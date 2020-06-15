#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize_scalar, minimize
import datetime
import os
import networkx as nx
import pandas as pd 
from gurobipy import*
from collections import Counter, defaultdict
from scipy.io import loadmat
from time import process_time
from matplotlib import ticker
import warnings
warnings.filterwarnings("ignore")
from autograd import grad as grad_a
from sklearn.preprocessing import StandardScaler


# # Oracles

# In[2]:



def line_search(x, d, gamma_max,func):
    
    #line-search using Brent's rule in scipy
    
    '''
    Minimizes f over [x, y], i.e., f(x+gamma*d) as a function of scalar gamma in [0,gamma_max]
    '''

    def fun(gamma):
        ls = x + gamma*d
        return func(ls)

    res = minimize_scalar(fun, bounds=(0, gamma_max), method='bounded')

    gamma = res.x
    ls = x + gamma*d        
    return ls, gamma


def segment_search(f, grad_f, x, y, tol=1e-6, stepsize=True):
    
    #line-search using golden-section rule coded from scratch
    
    '''
    Minimizes f over [x, y], i.e., f(x+gamma*(y-x)) as a function of scalar gamma in [0,1]
    '''
    
    # restrict segment of search to [x, y]
    d = (y-x).copy()
    left, right = x.copy(), y.copy()
    
    # if the minimum is at an endpoint
    if np.dot(d, grad_f(x))*np.dot(d, grad_f(y)) >= 0:
        if f(y) <= f(x):
            return y, 1
        else:
            return x, 0
    
    # apply golden-section method to segment
    gold = (1+np.sqrt(5))/2
    improv = np.inf
    while improv > tol:
        old_left, old_right = left, right
        new = left+(right-left)/(1+gold)
        probe = new+(right-new)/2
        if f(probe) <= f(new):
            left, right = new, right
        else:
            left, right = left, probe
        improv = np.linalg.norm(f(right)-f(old_right))+np.linalg.norm(f(left)-f(old_left))
    x_min = (left+right)/2
    
    # compute step size gamma
    gamma = 0
    if stepsize == True:
        for i in range(len(d)):
            if d[i] != 0:
                gamma = (x_min[i]-x[i])/d[i]
                break
                
    return x_min, gamma



def active_constraints_oracle(A,b,x):
    
    '''
    finding active constraints for polytope of the form Ax <= b
    '''
    b_prime = []
    A_prime = []
    A_not = []
    b_not = []
    for i in range(len(A)):
        if np.abs(np.round(np.dot(A[i],x),4) - b[i]) < 0.01:
            A_prime.append(list(A[i]))
            b_prime.append(b[i])
        else:
            A_not.append(list(A[i]))
            b_not.append(b[i])
            
    return np.array(A_prime),np.array(b_prime),np.array(A_not),np.array(b_not)


def max_step_size(A,b,x, d):
    
    
    '''
    finding maximum step-size: argmax{\delta | x + \delta d \in P} where P is given by Ax <= b and d is a 
    feasible direction at x. Here we use Gurobi as a black box solver.
    '''

    m = Model("opt")
    n = len(A.T)

    lam = m.addVar(lb=-GRB.INFINITY, name='lam')

    m.update()              

    objExp = lam
    m.setObjective(objExp, GRB.MAXIMIZE)
    
    m.update()

    #feasibility constraints
    
    for i in range(len(A)):
        m.addConstr(np.dot(np.array(x),A[i]) + lam* np.dot(np.array(d),A[i]),'<=', b[i])
        
    m.update()

    #optimize
    m.setParam( 'OutputFlag', False )
    m.optimize()
    
    return lam.x
    

def max_step_size_search(A,b,x, d):
    
    '''
    finding maximum step-size: argmax{\delta | x + \delta d \in P} where P is given by Ax <= b and d is a 
    feasible direction at x. Here we search over set of inactive constraints that might get violated.
    '''

    A_J,b_J = active_constraints_oracle(A,b,x)[2:]
    slack = b_J - np.dot(A_J,x)
    
    try:
        if len(d) > 1:
            denom = np.dot(A_J,d_pi)
            excess = [slack[i]/denom[i] for i in range(len(slack)) if denom[i] > 0]
            gamma_max = min(excess)
            return gamma_max
    except:
        return min(slack)


def shadow_oracle(A,b,x, d):
     
    '''
    Projecting d on set of feasible directions for polytope given by Ax <= b
    '''
    
    #find matrix of active constraints
    A_I = active_constraints_oracle(A,b,x)[0]

    m = Model("opt")
    n = len(A_I.T)

    z = []
    for i in range(n):
        z.append(m.addVar(lb=-GRB.INFINITY, name='z_{}'.format(i)))

    m.update()              

    objExp = quicksum(np.square(np.array([d[k] -z[k] for k in range(n)])))
   
    m.setObjective(objExp, GRB.MINIMIZE)
    m.update()

    #feasibility constraints
    for i in range(len(A_I)):
        m.addConstr(np.dot(np.array(z),A_I[i]),'<=', 0)
        
    m.update()

    #optimize
    m.setParam( 'OutputFlag', False )
    m.optimize()

    der = [i.x for i in z]

    return np.array(der)


def shadow_oracle_other(A,b,x, grad,M,vert_rep,vertices):
     
    '''
    Computing projection if we know an upper bound M on the value at which normal cone is not changing
    '''
    
    epsilon = M/2
    
    if vert_rep == True:
        g = projection_oracle_vertices(vertices,x - epsilon*grad)
        der = (g -x)/epsilon
        
    else:
        projection_oracle(x - epsilon*grad,A,b)
        der = (g -x)/epsilon

    return der

    return np.array(der)


# # Algorithms - FW Variants

# # FW

# In[3]:


def FW(x, lmo, epsilon,func,grad_f, f_tol, time_tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time = [0]
    f_improv = np.inf

    #initialize iteration count
    t = 0    

    while f_improv > f_tol and time[-1] < time_tol:
        
        start = process_time()
        
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        v = lmo(grad)
        d_FW = v-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))

        #Update next iterate by doing a feasible line-search
        x, gamma = segment_search(func, grad_f, x, v)
        
        end = process_time()
        
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        
        t+=1
        
    return x, function_value, time,t,primal_gap


# # AFW 

# In[4]:


#Function to compute away vertex
def away_step(grad, S):
    
    '''
    Compute away vertex by searching over current active set
    '''
    
    costs = {}
    
    for k,v in S.items():
        cost = np.dot(k,grad)
        costs[cost] = [k,v]
    vertex, alpha = costs[max(costs.keys())]  
    return vertex,alpha

#Function to update active set
def update_S(S,gamma, Away, vertex):
    
    '''
    Update convex decompistion of active step after every iteration 
    '''
    
    S = S.copy()
    vertex = tuple(vertex)
    
    if not Away:
        if vertex not in S.keys():
            S[vertex] = gamma
        else:
            S[vertex] *= (1-gamma)
            S[vertex] += gamma
            
        for k in S.keys():
            if k != vertex:
                S[k] *= (1-gamma)
    else:
        for k in S.keys():
            if k != vertex:
                S[k] *= (1+gamma)
            else:
                S[k] *= (1+gamma)
                S[k] -= gamma
    return {k:v for k,v in S.items() if np.round(v,3) > 0}


#AFW Algorithm
def AFW(x, lmo, epsilon,func,grad_f, f_tol, time_tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time = [0]
    f_improv = np.inf

    #initialize starting point and active set
    t = 0    
    S={tuple(x): 1}

    while f_improv > f_tol and time[-1] < time_tol:
        
        start = process_time()
        
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        v = lmo(grad)
        d_FW = v-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))

        #Compute away vertex and direction
        a,alpha_a = away_step(grad, S)
        d_A = x - a

        #Check if FW gap is greater than away gap
        if np.dot(-grad,d_FW) >= np.dot(-grad,d_A):
            #choose FW direction
            d = d_FW
            vertex = v
            gamma_max = 1
            Away = False
        else:
            #choose Away direction
            d = d_A
            vertex = a
            gamma_max = alpha_a/(1-alpha_a)
            Away = True

        #Update next iterate by doing a feasible line-search
        x, gamma = segment_search(func, grad_f, x, x + gamma_max *d)

        #update active set based on direction chosen
        S = update_S(S,gamma, Away, vertex)
        
        end = process_time()
        time.append(time[t] + end - start)
        
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        
        t+=1
        
    return x, function_value, time,t,primal_gap


# # Pairwise FW

# In[5]:


def update_S_PW(S,gamma, FW_vertex, Away_vertex):
    
    '''
    Update convex decompistion of active step after pairwise direction is chosen 
    '''
    
    vertex1 = tuple(FW_vertex)
    vertex2 = tuple(Away_vertex)
    S = S.copy()
    
    if vertex1 not in S.keys():
        S[vertex1] = gamma
    else:
        S[vertex1] += gamma
    
    S[vertex2] -= gamma
    return {k:v for k,v in S.items() if np.round(v,4) > 0}


#PFW Algorithm
def PFW(x, lmo, epsilon,func,grad_f, f_tol, time_tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time = [0]
    f_improv = np.inf

    #initialize starting point and active set
    t = 0    
    S={tuple(x): 1}

    while f_improv > f_tol and time[-1] < time_tol:
        
        start = process_time()
        
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        v = lmo(grad)
        d_FW = v-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))

        #Compute away vertex and direction
        a,alpha_a = away_step(grad, S)
        d_A = x - a

        #Pairwise step
        d =  d_FW + d_A
        gamma_max = alpha_a

        #Update next iterate by doing a feasible line-search
        x, gamma = line_search(x, d, gamma_max,func)

        #update active set
        S = update_S_PW(S,gamma, v, a)
        
        end = process_time()
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        
        t+=1
        
    return x, function_value, time,t,primal_gap


# # Decoposition Invaraint FW (DICG)

# In[6]:


#DICG algorithm
def DICG(x, lmo, feasibility_oracle, epsilon,func,grad_f, f_tol, time_tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time = [0]
    f_improv = np.inf

    #initialize starting point and active set
    t = 0    

    while f_improv > f_tol and time[-1] < time_tol:
        
        start = process_time()
        
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        v = lmo(grad)
        d_FW = v-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))

        #create new gradient to find best away vertex
        g = np.array([grad[i] if x[i] > 0 else -9e9 for i in range(len(x))])
        a = lmo(-g)
        d_A = x - a

        #Pairwise step
        d =  d_FW + d_A
        gamma_max = feasibility_oracle(x,d)

        #Update next iterate by doing a feasible line-search
        x, gamma = segment_search(func, grad_f, x, x+gamma_max*d)
        
        end = process_time()
        time.append(time[t] + end - start)
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        
        t+=1
        
    return x, function_value, time,t,primal_gap


# # Projected Gradient variants

# # Shadow Walk

# In[7]:


def trace_PW_curve(x,grad,shadow,feasibility_oracle,func,tol):
    
    '''
    trace the piecewise linear projection curve
    '''
    
    count = 0
    t = 0
    
    while True:
        #compute shadow of gradient
        d_pi = shadow(x,grad)
        t1 = process_time()
        
        #find maximum step size in which one can move along shadow
        gamma_max = feasibility_oracle(x, d_pi)
        
        #do line search alog shadow
        x, gamma = segment_search(func, grad_f, x, x + gamma_max *d_pi)
        
        #if optimal step is not maximal or we reached endpoint of curve - terminate
        if abs(gamma - 1) > 0.001 or np.dot(d_pi,d_pi)**0.5 < tol:
            break
        else:
            count +=1 #record breakpoints
        t2 = process_time()
        t += (t2 - t1)

    return x,count,t





#shadow descent algorithm:
def SD(x, lmo, shadow,feasibility_oracle, epsilon,func,grad_f, f_tol, time_tol,tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time1 = [0]
    time2 = [0]
    f_improv = np.inf
    counts = []

    #initialize starting point and active set
    t = 0    

    while f_improv > f_tol and time1[-1] < time_tol:
        
        start = process_time()
        
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        s = lmo(grad)
        d_FW = s-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))
        
        #Compute directional derivative
        t1 = process_time()
        d_pi = shadow(x,grad)
        t2 = process_time()
        
        gamma_max = feasibility_oracle(x, d_pi)

        #Update next iterate by doing a feasible line-search
        y, gamma = segment_search(func, grad_f, x, x + gamma_max *d_pi)
        
        #check for boundary case
        if abs(gamma - 1) < 0.001:
            
            trace = True
            
            #trace PW curve
            t3 = process_time()
            x, c, t_pw = trace_PW_curve(x,grad,shadow,feasibility_oracle,func,tol)
            counts.append(c)
            t4 = process_time()
            
        else:
            trace = False
            x = y
            counts.append(0)

        end = process_time()
        time1.append(time1[t] + end - start)

        if trace:
            time2.append( time2[t] + end - start - (t2 -t1) -(t4 - t3) + t_pw)
        else:
            time2.append( time2[t] + end - start - (t2 -t1))
            
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t+=1
        
    return x, function_value, time1,time2,t,primal_gap,counts


# # Shadow CG

# In[8]:


#shadow conditional gradients
def SCG(x, lmo, shadow,feasibility_oracle, epsilon,func,grad_f, f_tol, time_tol,tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time1 = [0]
    time2 = [0]
    f_improv = np.inf
    counts = []
    FW = []

    #initialize starting point and active set
    t = 0    

    while f_improv > f_tol and time1[-1] < time_tol:
        
        start = process_time()
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        v = lmo(grad)
        d_FW = v-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))
            
        #Compute directional derivative
        t1 = process_time()
        d_pi = shadow(x,grad)
        t2 = process_time()
        gamma_max = feasibility_oracle(x, d_pi)
        
        #Check if FW gap direction is better than normalized shadow
        if np.dot(-grad,d_FW) >= np.dot(-grad,d_pi/(np.dot(d_pi,d_pi)**0.5)):
            
            #wrap around using FW direction
            d = d_FW
            vertex = v
            gamma_max = 1
            
            #record which direction is chosen 
            shadow_dir = False
            FW.append(1) 
            
        else:
            
            #choose shadow direction
            d = d_pi
            
            #record which direction is chosen 
            shadow_dir = True
            FW.append(0) 

        #Update next iterate by doing a feasible line-search
        y, gamma = segment_search(func, grad_f, x, x + gamma_max*d)
        
        #check for boundary case in shadow steps
        if shadow_dir and abs(gamma - 1) < 0.001:
        
            #trace PW curve
            trace = True
            t3 = process_time()
            x, c, t_pw = trace_PW_curve(x,grad,shadow,feasibility_oracle,func,tol)
            t4 = process_time()
            counts.append(c)
            
        else:
            trace = False
            x= y
            counts.append(0)
            
        end = process_time()

        time1.append(time1[t] + end - start)

        if trace:
            time2.append( time2[t] + end - start - (t2 -t1) -(t4 - t3) + t_pw)
        else:
            time2.append( time2[t] + end - start - (t2 -t1))
            
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t+=1
        
    return x, function_value, time1,time2,t,primal_gap,counts,FW


# # PGD

# In[9]:


# PDG  algorithm:
def PGD(x, lmo,proj,L, epsilon,func,grad_f, f_tol, time_tol):
    
    #record primal gap, function value, and time every iteration
    now=datetime.datetime.now()
    primal_gap = []
    function_value=[func(x)]
    time = [0]
    f_improv = np.inf
    counts = []

    #initialize starting point and active set
    t = 0    

    while f_improv > f_tol and time[-1] < time_tol:
        #compute gradient
        grad = grad_f(x)

        #solve linear subproblem and compute FW direction
        s = lmo(grad)
        d_FW = s-x

        #If primal gap is small enough - terminate
        if np.dot(-grad,d_FW) <= epsilon:
            break
        else:
            #update convergence data
            primal_gap.append(np.dot(-grad,d_FW))

        # take a 1/L step-size in negative gradient direction and then project back
        x = proj(x,grad,L)
        
        later=datetime.datetime.now()
        time.append((later - now).total_seconds())
        f_improv = function_value[-1] - func(x)
        function_value.append(func(x))
        t+=1
        
    return x, function_value, time,t,primal_gap


# # Plotting

# In[10]:


styles = {'AFW':'-', 'PFW':'--', 'DICG':'-', 'Shadow':'-',
          'PGD':'--', 'Shadow CG':'-'}
colors = {'AFW':'tab:blue', 'PFW':'tab:blue', 'DICG':'tab:red', 'Shadow':'tab:green',
          'PGD':'tab:orange', 'Shadow CG':'tab:orange'}
labels = {'AFW':'AFW', 'PFW':'PFW', 'DICG':'DICG', 'PGD':'PGD',
          'Shadow':'Shadow Walk', 'Shadow CG':'Shadow CG'}


# In[11]:


def plotter(res, styles, colors, labels,oracle = False,fstar = None, log = True, sci = False, outfilename=None):
    
    '''
    plot primal gap specifying whether we want to assume oracle access for shadow or not
    '''
    
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)

    plt.subplot(1, 2, 1)
    
    if fstar:
        for alg in res.keys():
            plt.plot(np.arange(len(res[alg][1])), res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg],
                     label=labels[alg])
    else:
        for alg in res.keys():
            plt.plot(np.arange(len(res[alg][1])), res[alg][1], linestyle=styles[alg], color=colors[alg],
                     label=labels[alg])
    plt.legend()
    plt.ylabel(r'$\log (f(\mathbf{x}_t) - f(\mathbf{x}^*))$')
    plt.xlabel('Iteration')
    plt.xlim(0,60)
    plt.tick_params(which='minor', left=False)
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.grid(linestyle=':')

    plt.subplot(1, 2, 2)
    if fstar:
        for alg in res.keys():
            if oracle and alg in ['Shadow', 'Shadow CG']:
                plt.plot(res[alg][3], res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg])
                plt.xlabel('Wall-clock time assuming shadow oracle (s)')
            else:
                plt.xlabel('Wall-clock time (s)')
                plt.plot(res[alg][2], res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg])
    else:
        for alg in res.keys():
            
            if oracle and alg in ['Shadow', 'Shadow CG']:
                plt.plot(res[alg][3], res[alg][1], linestyle=styles[alg], color=colors[alg])
                plt.xlabel('Wall-clock time assuming shadow oracle (s)')
            else:
                plt.plot(res[alg][2], res[alg][1], linestyle=styles[alg], color=colors[alg])
                plt.xlabel('Wall-clock time (s)')
                
    plt.ylabel(r'$\log (f(\mathbf{x}_t) - f(\mathbf{x}^*))$')
    plt.tick_params(which='both', left=False, labelleft=False)
    plt.grid(linestyle=':')
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()


def plotter_gaps(res, styles, colors, labels,oracle = False, log = True, sci = False, outfilename=None):
    
    
    '''
    plot duality gap specifying whether we want to assume oracle access for shadow or not
    '''
    
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)

    plt.subplot(1, 2, 1)
    for alg in res.keys():
        if alg in ['Shadow', 'Shadow CG']:
            plt.plot(np.arange(len(res[alg][5])), res[alg][5], linestyle=styles[alg], color=colors[alg],
                     label=labels[alg])
        else:
            plt.plot(np.arange(len(res[alg][4])), res[alg][4], linestyle=styles[alg], color=colors[alg],
                     label=labels[alg])
    plt.legend()
    plt.ylabel(r'$\log(\mathrm{Duality}$' + ' gap)')
    plt.xlabel('Iteration')
    plt.tick_params(which='minor', left=False)
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.grid(linestyle=':')

    plt.subplot(1, 2, 2)
    for alg in res.keys():
        if oracle and alg in ['Shadow', 'Shadow CG']:
            plt.plot(res[alg][3][1:], res[alg][5], linestyle=styles[alg], color=colors[alg])
            plt.xlabel('Wall-clock time assuming shadow oracle (s)')
        elif alg in ['Shadow', 'Shadow CG']:
            plt.plot(res[alg][2][1:], res[alg][5], linestyle=styles[alg], color=colors[alg])
            plt.xlabel('Wall-clock time (s)')
        else:
            plt.plot(res[alg][2][1:], res[alg][4], linestyle=styles[alg], color=colors[alg])
            plt.xlabel('Wall-clock time (s)')
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.ylabel(r'$\log(\mathrm{Duality}$' + ' gap)')
    plt.tick_params(which='both', left=False, labelleft=False)
    plt.grid(linestyle=':')

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()

def plotter_break_point(res, styles, colors, labels, m = False,outfilename=None):
    
    '''
    plot number of oracle calls made
    '''
    
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    
    li = ['Shadow', 'Shadow CG']
    
    for alg in li:
        plt.plot(np.arange(len(res[alg][6])), np.array(res[alg][6]) + 1, linestyle=styles[alg], color=colors[alg],
                 label=labels[alg])
    if m:
        plt.axhline(m,  color='black', linestyle='--', label = r'$m$')
        

    plt.legend()
    #plt.ylabel('Number of Iterations spent in '+r'$\mathrm{Trace}(\mathbf{x}_t, \nabla f(\mathbf{x}_t))$')
    plt.ylabel('Number of calls to shadow oracle')
    plt.xlabel('Iteration')
    plt.tick_params(which='minor', left=False)
    plt.grid(linestyle=':')

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()

    
def plotter_cum(res, styles, colors, labels, m = False,outfilename=None):
    
    '''
    plot cumulative number of shadow steps taken
    '''
    
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    
    li = ['Shadow', 'Shadow CG']
    
    for alg in li:
        data = []
        for i in res[alg][6]:
            if i >= 1:
                data.append(i+1)
            else:
                data.append(i)
        if alg == 'Shadow':       
            plt.plot(np.arange(len(data)), np.cumsum(np.array(data)+1), linestyle=styles[alg], color=colors[alg],
                 label=labels[alg])
        else:      
            plt.plot(np.arange(len(data)), np.cumsum(np.array(data)), linestyle=styles[alg], color=colors[alg],
                 label=labels[alg])

    plt.legend()
    plt.ylabel('Cumulative number of shadow steps')
    plt.xlabel('Iteration')
    plt.tick_params(which='minor', left=False)
    plt.grid(linestyle=':')

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()

    
    
def plotter_wout_PGD(res, styles, colors, labels,oracle = False,fstar = None, log = True, sci = False, outfilename=None):
    
    
    '''
    plot primal gap without PGD -  specifying whether we want to assume oracle access for shadow or not
    '''
    
    
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)

    plt.subplot(1, 2, 1)
    
    if fstar:
        for alg in res.keys():
            if alg != 'PGD':
                plt.plot(np.arange(len(res[alg][1])), res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg],
                         label=labels[alg])
    else:
        for alg in res.keys():
            if alg != 'PGD':
                plt.plot(np.arange(len(res[alg][1])), res[alg][1], linestyle=styles[alg], color=colors[alg],
                     label=labels[alg])
    plt.legend()
    plt.ylabel(r'$\log (f(\mathbf{x}_t) - f(\mathbf{x}^*))$')
    plt.xlabel('Iteration')
    #plt.xlim(0,60)
    plt.tick_params(which='minor', left=False)
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.grid(linestyle=':')

    plt.subplot(1, 2, 2)
    if fstar:
        for alg in res.keys():
            if alg != 'PGD':
                if oracle and alg in ['Shadow', 'Shadow CG']:
                    plt.plot(res[alg][3], res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg])
                    plt.xlabel('Wall-clock time assuming shadow oracle (s)')
                else:
                    plt.xlabel('Wall-clock time (s)')
                    plt.plot(res[alg][2], res[alg][1] - fstar, linestyle=styles[alg], color=colors[alg])
    else:
        for alg in res.keys():
            if alg != 'PGD':
                if oracle and alg in ['Shadow', 'Shadow CG']:
                    plt.plot(res[alg][3], res[alg][1], linestyle=styles[alg], color=colors[alg])
                    plt.xlabel('Wall-clock time assuming shadow oracle (s)')
                else:
                    plt.xlabel('Wall-clock time (s)')
                    plt.plot(res[alg][2], res[alg][1], linestyle=styles[alg], color=colors[alg])

    plt.ylabel(r'$\log (f(\mathbf{x}_t) - f(\mathbf{x}^*))$')
    plt.tick_params(which='both', left=False, labelleft=False)
    plt.grid(linestyle=':')
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()



def plotter_gaps_wout_PGD(res, styles, colors, labels,oracle = False, log = True, sci = False, outfilename=None):
    
        
    '''
    plot duality gap without PGD -  specifying whether we want to assume oracle access for shadow or not
    '''
    
    
    fig = plt.figure()
    fig.set_figheight(3)
    fig.set_figwidth(10)

    plt.subplot(1, 2, 1)
    for alg in res.keys():
        if alg != 'PGD':
            if alg in ['Shadow', 'Shadow CG']:
                plt.plot(np.arange(len(res[alg][5])), res[alg][5], linestyle=styles[alg], color=colors[alg],
                         label=labels[alg])
            else:
                plt.plot(np.arange(len(res[alg][4])), res[alg][4], linestyle=styles[alg], color=colors[alg],
                         label=labels[alg])
    plt.legend()
    plt.ylabel(r'$\log(\mathrm{Duality}$' + ' gap)')
    plt.xlabel('Iteration')
    plt.tick_params(which='minor', left=False)
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.grid(linestyle=':')

    plt.subplot(1, 2, 2)
    for alg in res.keys():
        if alg != 'PGD':
            if oracle and alg in ['Shadow', 'Shadow CG']:
                plt.plot(res[alg][3][1:], res[alg][5], linestyle=styles[alg], color=colors[alg])
                plt.xlabel('Wall-clock time assuming shadow oracle (s)')
            elif alg in ['Shadow', 'Shadow CG']:
                plt.xlabel('Wall-clock time (s)')
                plt.plot(res[alg][2][1:], res[alg][5], linestyle=styles[alg], color=colors[alg])
            else:
                plt.xlabel('Wall-clock time (s)')
                plt.plot(res[alg][2][1:], res[alg][4], linestyle=styles[alg], color=colors[alg])
    if log:
        plt.yscale('log')
    if sci:
        plt.ticklabel_format(axis = 'y',style='sci',scilimits = (0,0))
    plt.ylabel(r'$\log(\mathrm{Duality}$' + ' gap)')
    plt.tick_params(which='both', left=False, labelleft=False)
    plt.grid(linestyle=':')

    if outfilename is not None:
        plt.savefig(outfilename, dpi=200, bbox_inches='tight')
        
    plt.show()


# # Shadow and Projection oracles using FW

# In[12]:


#projection oracle using FW
def proj_FW(x_0,y,lmo,f_tol,time_tol):
    
    '''
    use the FW algorithm to project y onto polytope P using x_0 as a starting point
    '''

    def f_proj(z):
        return np.dot(z - y,z - y)

    def grad_f_proj(z):
        return 2*(z - y)


    return FW(x_0, lmo, 0,f_proj,grad_f_proj, f_tol, time_tol)[0]


def shadow_oracle_FW_binary(x_0,grad_0,lmo,start,tol,f_tol,time_tol):
    
    
    '''
    use the FW algorithm to compute shadow of gradinet by doing binary search to find the breakpoint
    '''

    def f_proj(z):
        return np.dot(z - x_0 + epsilon*grad_0,z - x_0 + epsilon*grad_0)

    def grad_f_proj(z):
        return 2*(z - x_0 + epsilon*grad_0)

    epsilon = start

    while True:

        z1 = FW(x_0, lmo, 0,f_proj,grad_f_proj, f_tol, time_tol)[0]
        d_pi = (z1 - x_0)/epsilon

        epsilon = epsilon / 2
        z2 = FW(x_0, lmo, 0,f_proj,grad_f_proj, f_tol, time_tol)[0]

        if np.dot( np.round(x_0 + epsilon*d_pi -z2,5), np.round(x_0 + epsilon*d_pi - z2,5))**0.5  <= tol:
            break
        else:
            epsilon = epsilon/2

    return d_pi


# # Computational Experiments


# # Figures 11, 12 and 13: Lasso Regression (small instance)

# In[35]:


#specify problem dimension and noise parameter
m, n, sigma = 40, 60, 0.1
np.random.seed(4)

#generate problem data
x_star = np.zeros(n)
rand = np.random.randint(1,n,int(0.25*n))
for i in rand[:int(len(rand)*0.5)]:
    x_star[i] = 1
    
for i in rand[:int(len(rand)*0.5)]:
    x_star[i] = -1

A = np.random.normal(0.5,0.25,(m, n))
y = np.dot(A, x_star)+sigma*np.random.randn(m)


#define objective function and gradient
f = lambda z: np.linalg.norm(y-np.dot(A, z[:n]-z[n:]), 2)**2

def grad_f(x):
    return np.concatenate([2*np.matmul(A.T,np.dot(A, x[:n]-x[n:]) - y),-2*np.matmul(A.T,np.dot(A, x[:n]-x[n:]) - y)])


#define linear optimization oracle
tau = np.linalg.norm(x_star, 1)
V = tau*np.identity(2*n)
x = V[np.random.randint(len(V))].copy()
lmo = lambda g: V[np.argmin(g)]
    
#generate constraint matrix:
B = np.vstack((np.ones(2*n), -np.ones(2*n), -np.eye(2*n)))
e = [tau] + [-tau] + [0]*(2*n)
#polytope is now given by Bx <= e

#choose desired feasibility oracle 
#since our polytope is a scaled simplex, the feasibility oracle could simply be the following
def feasibility_oracle (x,d):
    eta = np.min([-x[i]/d[i] if d[i] < 0 else np.inf for i in range(len(x))])
    eta = min(eta, 1)
    return eta

#choose desired shadow oracle - here for example we can gurobi to compute shadow
def shadow(x,d):
    return shadow_oracle(B,e,x, -d)

f_tol,time_tol,epsilon,tol = 0.00001, np.inf, 0.1, 0.001


# In[18]:


res_lasso_sparse_small = {}
print('AFW')
res_lasso_sparse_small['AFW'] = AFW(x, lmo, epsilon,f,grad_f, f_tol, time_tol)
print('PFW')
res_lasso_sparse_small['PFW'] = PFW(x, lmo, epsilon,f,grad_f, f_tol, time_tol=np.inf)
print('DICG')
res_lasso_sparse_small['DICG'] = DICG(x, lmo, feasibility_oracle, epsilon,f,grad_f, f_tol, time_tol)
print('Shadow')
res_lasso_sparse_small['Shadow'] = SD(x, lmo, shadow,feasibility_oracle, epsilon,f,grad_f, f_tol, time_tol,tol)
print('Shadow CG')
res_lasso_sparse_small['Shadow CG'] = SCG(x, lmo, shadow,feasibility_oracle, epsilon,f,grad_f, f_tol, time_tol,tol)


# In[36]:


f_star = f(res_lasso_sparse_small['AFW'][0])

#plot primal gaps and duality gaps without assuming oracle access to shadow
plotter_wout_PGD(res_lasso_sparse_small, styles, colors, labels, False, f_star, log = True)
plotter_gaps_wout_PGD(res_lasso_sparse_small, styles, colors, labels, False,log = True)

#plot primal gaps and duality gaps assuming oracle access to shadow and leaving PGD out for a better comparison
plotter_wout_PGD(res_lasso_sparse_small, styles, colors, labels, True, f_star, log = True)
plotter_gaps_wout_PGD(res_lasso_sparse_small, styles, colors, labels, True,log = True)

#plot number of shadow oracle calls made and access cumulative number of shadow steps taken by Shadow-CG and Shadow-Walk
plotter_break_point(res_lasso_sparse_small, styles, colors, labels)
plotter_cum(res_lasso_sparse_small, styles, colors, labels)


