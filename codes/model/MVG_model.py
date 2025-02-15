# Mualem-van Genuchen model
'model info '
theta_r = 0.078
theta_s = 0.43
alpha = 0.036
n = 1.56

def theta(psi):
    m = 1 - 1 / n
    S_e = (1 + (alpha*abs(psi))**n)**(-m)
    return S_e * (theta_s - theta_r) + theta_r

def Kr(theta):
    m = 1 - 1 / n
    S_e = (theta - theta_r) / (theta_s - theta_r)
    return S_e**0.5*(1-(1-S_e**(1/m))**m)**2

def K(Ks, Kr):
    return Ks*Kr