"""
This code generates samples of the numerical approximations to the quasilinear SDE from Section 5. 
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

# Generate sample of fBm of N steps with step size dt
def fBm_sample(H,N,dt):
    a=fBm_autocov(np.arange(N),H,dt)
    d=circ_embedding(a)
    I1,I2=fBm_increments(d,N)  
    B1=np.cumsum(I1)
    B2=np.cumsum(I2)  
    return(np.concatenate(([0],B1)),np.concatenate(([0],B2)))
    

## Update steps
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
    
# Function that determines the approximation X via the process Z. It needs the underlying fBm B as input. The function Afunc determines the value of A in the Rosenbrock approximation.    
def Z_method(B,H,T,N,alpha,beta,x0,func,Afunc):
    # Calculate some useful constants
    dt=T/N
    bB=-beta*B
    const=0.5*beta**2*dt**(2*H)
    c=const*np.arange(N+1)**(2*H)    
    
    # Lists that track the value of X for the different methods 
    X_Mishura=np.zeros(N+1)
    X_EF=np.zeros(N+1)
    X_RB=np.zeros(N+1)
    X_Mishura[0]=x0
    X_EF[0]=x0
    X_RB[0]=x0
    
    # Determine X_n for each n
    for n in range (1,N+1):
        Z_Mishura=x0
        Z_EF=x0
        Z_RB=x0

        # Run the underlying approximation for Z_i
        for i in range (0,n):
            J=np.exp(-alpha*i*dt+bB[i]+c[n]-const*(n-i)**(2*H))
            
            Z_Mishura=Mishura_step(Z_Mishura,J,dt,i,func)
            Z_EF=EF_step(Z_EF,J,dt,i,func)
            Z_RB=RB_step(Z_RB,J,dt,i,func,Afunc)

        J=np.exp(-alpha*n*dt+bB[n]+c[n])
        X_Mishura[n]=Z_Mishura/J
        X_EF[n]=Z_EF/J
        X_RB[n]=Z_RB/J

    return(X_Mishura,X_EF,X_RB)
    
# Function that determines the approximation X for the either the Euler-Maruyama or the heuristic method. It needs the underlying fBm B as input. It also needs an indication of which method is to be used,
def X_method(B,H,T,N,alpha,beta,x0,func,method):
    dt=T/N
    X=np.zeros(N+1)
    X[0]=x0
    
    # Calculate X iteratively
    for i in range (0,N):
        X[i+1]=method(H,B[i+1]-B[i],X[i],dt,i,func,alpha,beta)
        
    return(X)

## Plot approximations
""" 
This part plots a sample of the numerical approximation to the quasilinear SDE for different numerical methods.
"""

# Set nonlinear drift funtion a(t,x)
def a(t,x):
    return(4*x/(1+x**2))

# Set function for Rosenbrock approximation. Usually the derivative with respect to x of a(t,x).    
def A(t,x):
    return(4*(x**2-1)/(1+x**2)**2)  

# Set parameters
H=0.75
N=10**3            # Amount of steps
T=2                 # Final time            
dt=T/N              # Calculate step size  

alpha=-1
beta=1
x0=1                # Initial value    
func=a
Afunc=A

# Toggles for which methods we want to plot
Mishura=True            # Mishura method from Section 5.3
EF=False                 # Exponential freeze method from Section 5.5 
RB=False                 # Rosenbrock type method from Section 5.5
heuristic=True	        # Heuristic method form Section 5.5
EM=False                 # The (naive) Euler-Maruyama method 
gfBm_exact=False         # The exact gfBm (for a(t,x)=0)
          
# Generate underlying fBm
B=fBm_sample(H,N,dt)[0]

# Generate some of the approximations
 
X_Mishura,X_EF,X_RB=Z_method(B,H,T,N,alpha,beta,x0,func,Afunc)

# Generate some of the approximations and plot them
opacity=1               # Determines opacity of plot
t=np.linspace(0,T,N+1) 
if Mishura:
    plt.plot(t,X_Mishura,label='Mishura',alpha=opacity,linestyle='-')
if EF:    
    plt.plot(t,X_EF,label='Exponential freeze',alpha=opacity,linestyle='--')
if RB:    
    plt.plot(t,X_RB,label='Rosenbrock',alpha=opacity,linestyle=':')      

if heuristic:
    X=X_method(B,H,T,N,alpha,beta,x0,func,Heuristic_step)
    plt.plot(t,X,label='Heuristic method',alpha=opacity,linestyle='--') 
if EM:
    X=X_method(B,H,T,N,alpha,beta,x0,func,EM_step)
    plt.plot(t,X,label='EM',alpha=opacity,linestyle=':') 
    
if gfBm_exact:
    X=x0*np.exp(alpha*t-1/2*beta**2*t**(2*H)+beta*B)   
    plt.plot(t,X,label='Exact gfBm',alpha=opacity,linestyle='-.')
    
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.legend()
plt.title('Approximations of solution for $H=${}'.format(H))
plt.show()