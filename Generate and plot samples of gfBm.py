"""
This code generates samples of the gfBm and plots them in several ways.
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

# Generate sample of gfBm of N steps with step size dt   
def gfBm_sample(H,N,dt,alpha,beta,x0):
    B1,B2=fBm_sample(H,N,dt)    
    t=np.linspace(0,T,N+1)
    X1=x0*np.exp(alpha*t-1/2*beta**2*t**(2*H)+beta*B1)
    X2=x0*np.exp(alpha*t-1/2*beta**2*t**(2*H)+beta*B2)
    return(X1,X2)


## Plot two samples of the gfBm   
""" 
This part plots two sample paths of the geometric fBm.
"""

# Set parameters
H=0.75
N=10**6             # Amount of steps
T=1                 # Final time            
dt=T/N              # Calculate step size

alpha=1
beta=1
x0=1                # Initial value
        
# Generate samples        
X,Y=gfBm_sample(H,N,dt,alpha,beta,x0)

# Plot samples
t=np.linspace(0,T,N+1)
plt.plot(t,X)
plt.plot(t,Y)
plt.xlabel('$t$')
plt.ylabel('$B^H(t)$')
plt.title('Two geometric fBms, $H={}$'.format(H))
plt.show()


## Plot for several values of H (single plot, common random numbers)
"""
This part plots a sample path of the geometric fBm for several values of H.
All paths are in the same figure.
The paths are generated using common random numbers.
"""

# List of H values
H=np.array([0.1,0.25,0.5,0.75,0.9])

# Set parameters
N=10**5            # Amount of steps
T=10                 # Final time            
dt=T/N              # Calculate step size 

alpha=0.25
beta=0.1
x0=1                # Initial value

# Generate and plot samples
seed=np.random.randint(1000)
t=np.linspace(0,T,N+1)
for i in range (0,len(H)):
    np.random.seed(seed)   # Set seed for common random numbers
    X,Y=gfBm_sample(H[i],N,dt,alpha,beta,x0)
    plt.plot(t,X,label='$H=$ '+str(H[i]),linewidth=2*H[i]) 
plt.xlabel('$t$')
plt.legend()
plt.title('Some geometric fBms for varying values of $H$')
plt.show()    


## Refinement subplots
"""
This part plots refinements of the same gfBm in several subplots.
"""

# Set parameters
H=0.25
N=10**5             # Amount of steps in finest grid
T=10                 # Final time            
dt=T/N              # Calculate step size
refinements=4       # Amount of refinements (needs to be even)    

alpha=0.25
beta=0.1
x0=1                # Initial value

# Generate fBm sample on the finest grid
B=fBm_sample(H,N,dt)[0]

# Plot refinements
for i in range (0,refinements):
    # Get the fBm on the right grid    
    Brefine=B[0:N:10**i]
    
    t=np.linspace(0,T,int(N/10**i))
    X=x0*np.exp(alpha*t-1/2*beta**2*t**(2*H)+beta*Brefine)
    
    # Plot the gfBm
    plt.subplot(int(refinements/2),2,int(refinements-i))
    plt.plot(t,X,label='gfBm')
    plt.plot(t,x0*np.exp(alpha*t),label='Expectation',linestyle='--')
    plt.legend()
    plt.xlabel('$t$')
    plt.title('$N={}$'.format(int(N/10**i)))
plt.subplots_adjust(hspace=0.5)
plt.show()