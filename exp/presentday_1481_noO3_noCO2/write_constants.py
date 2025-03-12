import os
import filecmp

def write_constants_fn(radius,omega,grav,rdgas,kappa,wtmair):
    os.system('cp srcmods/constants.F90 srcmods/constants_temp.F90')

    with open('srcmods/constants_temp.F90', 'r') as file:
        data = file.readlines()


    data[58] = 'real, public, parameter :: RADIUS = '+str(radius)+'\n'
    data[59] = 'real, public, parameter :: OMEGA  = '+str(omega)+'\n'
    data[60] = 'real, public, parameter :: GRAV   = '+str(grav)+'\n'
    data[61] = 'real, public, parameter :: RDGAS  = '+str(rdgas)+'\n'
    data[62] = 'real, public, parameter :: KAPPA  = '+str(kappa)+'\n'
    data[143] = 'real, public, parameter :: WTMAIR = '+str(wtmair)+'\n'
    
    with open('srcmods/constants_temp.F90', 'w') as file:
        file.writelines( data )


    no_change = filecmp.cmp('srcmods/constants_temp.F90', 'srcmods/constants.F90')

    if no_change == True:
        os.system('rm srcmods/constants_temp.F90')
        print('No change to constants')
    else:
        os.system('mv srcmods/constants_temp.F90 srcmods/constants.F90')
        print('Constants updated, recompiling whole model')
