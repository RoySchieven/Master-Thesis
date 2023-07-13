"""
This code generates plots of the naive integral approach discussed in Section 2.6.
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
    
# Compute the Ito sum    
def naive_integral(H,N,dt):
    B1,B2=fBm_sample(H,N,dt)
    I1=B1[1:N+1]-B1[0:N]
    I2=B2[1:N+1]-B2[0:N]
    return(np.cumsum(B1[0:N]*I1),np.cumsum(B2[0:N]*I2))
    
    
## Plot two samples of the naive integral
""" 
This part plots two samples of the naive integral.
"""

# Set parameters
H=0.75
N=10**6             # Amount of steps
T=1                 # Final time            
dt=T/N              # Calculate step size
        
# Generate samples        
X,Y=naive_integral(H,N,dt)

# Plot samples
t=np.linspace(0,T,N)
plt.plot(t,X)
plt.plot(t,Y)
plt.xlabel('$t$')
plt.ylabel('Integral')
plt.title('Two fBms integrated against themselves, $H={}$'.format(H))    
plt.show()


## Refinement subplots
"""
This part plots grid refinements of the same naive integral in several subplots.
"""

# Set parameters
H=0.75
N=10**5             # Amount of steps for the finest grid
T=1                 # Final time            
dt=T/N              # Calculate step size
refinements=4       # Amount of refinements (needs to be even)    

# Generate fBm sample on the finest grid
B=fBm_sample(H,N,dt)[0]

# Plot refinements
for i in range (0,refinements):
    # Get the fBm and its increments on the right grid
    Brefine=B[0:N:10**i]
    Irefine=B[10**i:N+1:10**i]-B[0:N:10**i]
    
    # Plot the integral
    plt.subplot(int(refinements/2),2,int(refinements-i))
    plt.plot(np.linspace(0,T,int(N/10**i)),np.cumsum(Brefine*Irefine))
    plt.xlabel('$t$')
    plt.title('$N={}$'.format(int(N/10**i)))
plt.subplots_adjust(hspace=0.5)
plt.show()