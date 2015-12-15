import numpy as np


# Ornstein-Uhlenbeck process as Euler-Maruyama integral, ie with indepdent Gaussian increments.
def ou(N, theta = 1, mu = 0, sigma = 1, dt = 1):

    x = np.zeros(N);
    # x[0] = N0; 
    x[0] = dt*theta*mu + np.sqrt(dt)*sigma*np.random.randn(); 

    for i in xrange(1,N):
        x[i] = x[i-1] + dt*(theta*(mu-x[i-1])) + np.sqrt(dt)*sigma*np.random.randn();
    return x;    

