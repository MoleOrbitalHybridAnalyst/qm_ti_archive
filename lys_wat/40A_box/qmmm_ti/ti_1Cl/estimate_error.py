import numpy as np
from statsmodels.tsa import ar_model
from sys import argv

def sample_variance(noise_sigma, rho, n):
    var = noise_sigma**2 / (1 - rho**2)
    var /= (n**2)
    var *= (n + 2 * n * rho / (1 - rho) + 2 * (rho**n - 1) / (1 - rho)**2 * rho)
    return var

def print_res(ts, res):
    delta_, rho_ = res.params
    print("estimated mean:", delta_ / (1 - rho_))
    print("sample mean:", np.mean(ts))
    print("estimated rho:", rho_)
    print("estimated variance:", res.sigma2 / (1 - rho_**2))
    sample_var = sample_variance(np.sqrt(res.sigma2), rho_, len(ts))
    print("estimated sample variance:", sample_var)
    return sample_var

if __name__ == '__main__':
    ts = np.loadtxt(argv[1])
    N = len(ts)

    # try to minimize sample variance by removing some initial data
    minvar = np.inf
    min_t = None
    stepsize = 10
    for i in range(max(1, N // stepsize // 2)):
        m = ar_model.AutoReg(ts[i*stepsize:], 1, seasonal=False)
        res = m.fit()
        print("----------------------------------------------")
        print(f"using data[{i*stepsize}:]")
        print("----------------------------------------------")
        svar = print_res(ts[i*stepsize:], res)
        print("")
        
        if svar < minvar:
            minvar = svar
            min_t = i

    print("----------------------------------------------")
    print(f"min var found with data[{min_t*stepsize}:]")
    print("----------------------------------------------")
    m = ar_model.AutoReg(ts[min_t*stepsize:], 1, seasonal=False)
    res = m.fit()
    print_res(ts[min_t*stepsize:], res)
