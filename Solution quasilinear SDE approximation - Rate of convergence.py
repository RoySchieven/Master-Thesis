"""
This code estimates the rate of convergence of the numerical approximations to the quasilinear SDE from Section 5. 
"""

import numpy as np
import matplotlib.pyplot as plt


## Functions to sample fBm
"""
These funcions generates sample paths of the fBm as discussed in Appendix A.
"""

# Autocorrelation function of increments of fBm of size dt 
def fBm_autocov(n,H,dt):
    return(0.5*dt**(2*H)*(abs(n+1)**(2*H)+abs(n-1)**(2*H)-2*abs(n)**(2*H)))

# Generate circulant embedding vector and obtain diagonal d
def circ_embedding(a):
    atilde=np.concatenate([a,a[len(a)-2:0:-1]])
    d=len(atilde)*np.fft.ifft(atilde)
    return(d)

# Use the diagonal d to generate two samples of fBm increments   
def fBm_increments(d,N):
    K=len(d)
    xi=np.random.normal(size=K)+np.random.normal(size=K)*1j
    Z=(np.fft.fft(np.sqrt(d)*xi)/np.sqrt(K))[0:N]
    return(np.real(Z),np.imag(Z))  

# Generate sample of fBm with the diagonal d as input (prevents the need to calculate the diagonal multple times)
def fBm_sample_multiple(d,N):
    I1,I2=fBm_increments(d,N)  
    B1=np.cumsum(I1)
    B2=np.cumsum(I2)  
    return(np.concatenate(([0],B1)),np.concatenate(([0],B2)))
    

## Update steps for different methods
"""
These funcions perform one iterative step for Z or X.
"""

# Update steps for Z for the Mishura method, the exponential freeze method and the Rosenbrock method
def Mishura_step(Z,J,dt,i,func):
    return(Z+dt*J*(func(i*dt,Z/J)))
    
def EF_step(Z,J,dt,i,func):
    return(np.exp(dt*func(i*dt,Z/J)/(Z/J))*Z)
    
def RB_step(Z,J,dt,i,func,Afunc):
    return(Z*np.exp(dt*Afunc(i*dt,Z))+dt*(J*func(i*dt,Z/J)-Z*Afunc(i*dt,Z)))

# Update steps for X for Euler-Maruyama and the heuristic method
def EM_step(H,I,X,dt,i,func,alpha,beta):
    return(X*(1+alpha*dt+beta*I)+dt*func(i*dt,X))
    
def Heuristic_step(H,I,X,dt,i,func,alpha,beta):
    return(np.exp(alpha*dt-0.5*beta**2*dt**(2*H)*((i+1)**(2*H)-i**(2*H))+beta*I)*(X+dt*func(i*dt,X)))
    
# Function that determines the approximation X via the process Z. It needs the underlying fBm B as input. It also needs an indication of which method is to be used, alongside extra arguments for that method.    
def Z_method(B,H,T,N,alpha,beta,x0,func,method,*args):
    # Calculate some useful constants
    dt=T/N
    const=0.5*beta**2*dt**(2*H)
    
    # Run the underlying approximation for Z_n
    Z=x0
    for i in range (0,N):
        J=np.exp(-alpha*i*dt-beta*B[i]+const*(N**(2*H)-(N-i)**(2*H)))
        Z=method(Z,J,dt,i,func,*args)
    J=np.exp(-alpha*T-beta*B[N]+const*N**(2*H))
    X=Z/J   
    
    return(X)
    
# Function that determines the approximation X for the either the Euler-Maruyama or the heuristic method. It needs the underlying fBm B as input. It also needs an indication of which method is to be used,
def X_method(B,H,T,N,alpha,beta,x0,func,method):
    dt=T/N
    X=x0
    
    # Calculate X iteratively
    for i in range (0,N):
        X=method(H,B[i+1]-B[i],X,dt,i,func,alpha,beta)
    
    return(X)


## Monte Carlo function
"""
This function estimates the RMSE of the numerical methods by using a Monte Carlo method. The input Nref is the amount of steps taken for the reference solution, M the Monte Carlo sample size, Nstart the amount of steps in the coarsest mesh (the amount of steps will increase with factor 10 each data point), datapoints the amount of different step sizes and method the numerical method to be used.
"""

def MC_method(H,T,alpha,beta,x0,func,Nref,M,Nstart,datapoints,method,*args):
    dt_ref=T/Nref
    
    # Determine whether to work via Z or X
    if method==Mishura_step or method==EF_step or method==RB_step:
        process=Z_method
    else:
        process=X_method
    
    # Generate diagonal d        
    a=fBm_autocov(np.arange(Nref),H,dt_ref)
    d=circ_embedding(a)

    # Variable that tracks the sum of square errors
    S=np.zeros(datapoints)  
    
    for i in range (0,int(M/2)):
        # Generating reference fBm
        Bref1,Bref2=fBm_sample_multiple(d,Nref)
        
        # Determine reference solution (use Mishura method as reference)
        Xref1=Z_method(Bref1,H,T,Nref,alpha,beta,x0,func,Mishura_step)
        Xref2=Z_method(Bref2,H,T,Nref,alpha,beta,x0,func,Mishura_step)
        
        for j in range (0,datapoints):
            N=Nstart*10**j
            m=int(Nref/N)  # Conversion factor to reduce reference fBm
            
            # Create fBm on coarser grid
            B1=np.zeros(N+1)
            B2=np.zeros(N+1)
            for k in range (0,N+1):
                B1[k]=Bref1[m*k]
                B2[k]=Bref2[m*k]
            
            # Run numerical method
            X1=process(B1,H,T,N,alpha,beta,x0,func,method,*args)
            X2=process(B2,H,T,N,alpha,beta,x0,func,method,*args)
            
            # Update sum of squares
            S[j]+=(Xref1-X1)**2+(Xref2-X2)**2
            
    return(np.sqrt(S/M))


## Plot the estimated RMSE
""" 
This part plots determines and plots the estimated RMSE of the quasilinear SDE.
"""

# Set non-linear drift funtion a(t,x)
def a(t,x):
    return(4*x/(1+x**2))

# Set function for Rosenbrock approximation. Usually the derivative with respect to x of a(t,x).    
def A(t,x):
    return(4*(x**2-1)/(1+x**2)**2)  

# Set parameters
H=0.25
T=2                 # Final time            

alpha=-1
beta=1
x0=1                # Initial value    
func=a
Afunc=A

M=100               # Monte Carlo sample size
Nref=10**6          # Amount of steps for reference solution
Nstart=10           # The biggest amount of steps considered, all others will be factor 10^i larger
datapoints=4        # The amount of datapoints, i.e. N=Nstart,10*Nstart,...,10^(datapoints-1)*Nstart are the considered amount of steps

method=Mishura_step  # Method used for approximation 
    
# Obtaining the log of the estimated RMSE
if method==RB_step:
    logX=np.log(MC_method(H,T,alpha,beta,x0,func,Nref,M,Nstart,datapoints,method,A))
else:
    logX=np.log(MC_method(H,T,alpha,beta,x0,func,Nref,M,Nstart,datapoints,method))

# The log for the dt of the datapoints
logdt=np.zeros(datapoints)
for i in range (0,datapoints):
    logdt[i]=np.log(T/(10**i*Nstart))
 
# Fit a linear function through the datapoints
fitlogt=np.linspace(logdt[datapoints-1],logdt[0],10)
a,b=np.polyfit(logdt,logX,1)
fitlog=a*fitlogt+b

# Plot the results
plt.scatter(logdt,logX,label='Error')
plt.plot(fitlogt,fitlog,label='Linear fit, slope={:.3f}'.format(a),linestyle='--')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel('log $\Delta t$')
plt.ylabel('log RMSE')
plt.title('RMSE for $H=${}'.format(H))
plt.show()

print('The estimated rate of convergence is {:.3f}'.format(a))