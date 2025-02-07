# p/p0 = exp[-scale_heights*(surf_res*x + (1-surf_res)*x^exponent)]

import numpy as np
import matplotlib.pyplot as plt

isca_plots = '/proj/bolinc/users/x_ryabo/Isca_plots'

dirs = {
    "plot_output": isca_plots+"/"
    }

save_figs = False

surf_res      = 0.06 #0.075 #0.2
scale_heights = 7.0 #7.6 #11.0 
exponent      = 3.0 #4.0 #7.0 
x = np.arange(0,1.01,0.01)
p0 = 1e5 # Reference pressure in Pa

sigma = np.exp( -scale_heights*(surf_res*x + (1.-surf_res)*x**exponent) )
p = sigma*p0 # pressure array


fig, ax = plt.subplots(dpi=150)      
y = sigma

ax.semilogy(x,y,marker='x')
ax.invert_yaxis()
ax.set_ylabel("Sigma pressure")
#ax.set_ylabel(r"Pressure, $P \: [\mathrm{Pa}]$")

plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.ylim([y[-1],y[0]])

if save_figs:
    plt.savefig(dirs["plot_output"]+'sigma_levels.pdf',bbox_inches='tight')
    plt.savefig(dirs["plot_output"]+'sigma_levels.png',bbox_inches='tight')

# Show figure
plt.show()
#plt.close()