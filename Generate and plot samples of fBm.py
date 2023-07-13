"""
This code generates samples of the fBm and plots them in several ways.
"""

import numpy as np
import matplotlib.pyplot as plt


## Functions to sample fBm
"""
These funcions generate two sample paths of the fBm as discussed in Appendix A.
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
    
    
## Plot two samples of the fBm  
""" 
This part plots two sample paths of the fBm.
"""

# Set parameters
H=0.75
N=10**6             # Amount of steps
T=1                 # Final time            
dt=T/N              # Calculate step size
        
# Generate samples        
X,Y=fBm_sample(H,N,dt)

# Plot
t=np.linspace(0,T,N+1)
plt.plot(t,X)
plt.plot(t,Y)
plt.xlabel('$t$')
plt.ylabel('$B^H(t)$')
plt.title('Fractional Brownian motions, $H={}$'.format(H))
plt.show()


## Plot for several values of H (separate plots, no common random numbers)
"""
This part plots two sample paths of the fBm for several values of H.
Each value of H has its own figure. 
The paths are not generated using common random numbers.
"""

# List of H values (must have even length)
H=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# Set parameters
N=10**5            # Amount of steps
T=1                 # Final time            
dt=T/N              # Calculate step size 

# Generate and plot samples
t=np.linspace(0,T,N+1)
n=int(len(H)/2)
for i in range (0,len(H)):
    X,Y=fBm_sample(H[i],N,dt)
    plt.subplot(n,2,i+1)
    plt.plot(t,X) 
    plt.plot(t,Y)     
    plt.xlabel('$t$')
    plt.ylabel('$B^H(t)$')
    plt.title('$H={}$'.format(H[i]))
plt.subplots_adjust(hspace=2.0)
plt.show()    


## Plot for several values of H (single plot, no common random numbers)
"""
This part plots a sample path of the fBm for several values of H.
All paths are in the same figure.
The paths are not generated using common random numbers.
"""

# List of H values
H=np.array([0.1,0.25,0.5,0.75,0.9])

# Set parameters
N=10**5            # Amount of steps
T=10                 # Final time            
dt=T/N              # Calculate step size 

# Generate and plot samples
t=np.linspace(0,T,N+1)
for i in range (0,len(H)):
    X,Y=fBm_sample(H[i],N,dt)
    plt.plot(t,X,label='$H=$ '+str(H[i]),linewidth=2*H[i])
plt.xlabel('$t$')
plt.ylabel('$B^H(t)$')
plt.legend()
plt.title('Some fBms for varying values of $H$')
plt.show()    


## Plot for several values of H (single plot, common random numbers)
"""
This part plots a sample path of the fBm for several values of H.
All paths are in the same figure.
The paths are generated using common random numbers.
"""

# List of H values
H=np.array([0.1,0.25,0.5,0.75,0.9])

# Set parameters
N=10**5            # Amount of steps
T=10                 # Final time            
dt=T/N              # Calculate step size 

# Generate and plot samples
seed=np.random.randint(1000)
t=np.linspace(0,T,N+1)
for i in range (0,len(H)):
    np.random.seed(seed)   # Set seed for common random numbers
    X,Y=fBm_sample(H[i],N,dt)
    plt.plot(t,X,label='$H=$ '+str(H[i]),linewidth=2*H[i]) 
plt.xlabel('$t$')
plt.ylabel('$B^H(t)$')
plt.legend()
plt.title('Some fBms for varying values of $H$')
plt.show()    


## Plot multiple samples of same H
"""
This part plots multiple sample paths of the fBm for the same value of H.
All paths are in the same figure.
"""

# Generate sample of fBm with the diagonal d as input (prevents the need to calculate the diagonal multple times)
def fBm_sample_multiple(d,N):
    I1,I2=fBm_increments(d,N)  
    B1=np.cumsum(I1)
    B2=np.cumsum(I2)  
    return(np.concatenate(([0],B1)),np.concatenate(([0],B2)))

# Set parameters
H=0.8
N=10**5            # Amount of steps
T=1                 # Final time            
dt=T/N              # Calculate step size 
M=10                # Amount of paths (needs to be even)

# Generate diagonal d        
a=fBm_autocov(np.arange(N),H,dt)
d=circ_embedding(a)
    
# Generate and plot samples
t=np.linspace(0,T,N+1)
for i in range (0,int(M/2)):
    X,Y=fBm_sample_multiple(d,N)
    plt.plot(t,X)
    plt.plot(t,Y)
plt.xlabel('$t$')
plt.ylabel('$B^H(t)$')
plt.title('Fractional Brownian motions, $H={}$'.format(H))
plt.show()

    
## Simple Monte Carlo test
"""
This part performs a simple Monte Carlo routine to estimate the expectation and variance of B^H(T) and compares them to the theoretical values of 0 and T^(2H), respectively.
It serves as a quick check of our numerical sample method.
"""

# Set parameters
H=0.75
N=10**5             # Amount of steps
T=10                 # Final time            
dt=T/N              # Calculate step size
M=1000              # Monte Carlo sample size (needs to be even)

# Obtain samples
final_values=np.zeros(M)
for i in range (0,int(M/2)):
    X,Y=fBm_sample(H,N,dt)
    final_values[2*i]=X[-1]
    final_values[2*i+1]=Y[-1]

# Print results   
print('The sample mean is {:.3f}, while the exact expectation is {}'.format(np.mean(final_values),0))    
print('The sample variance is {:.3f}, while the exact variance is {:.3f}'.format(np.var(final_values),T**(2*H)))   