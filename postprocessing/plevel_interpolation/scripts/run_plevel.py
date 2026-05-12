from netCDF4 import Dataset  
from plevel_fn import plevel_call, daily_average, join_files, two_daily_average, monthly_average
import sys
import os
import time
import pdb
import subprocess

start_time=time.time()
base_dir='/proj/bolinc/users/x_ryabo/Isca-Ryan_outputs'
#exp_name_list = ['soc_ga3_files_smooth_topo_fftw_mk1_fresh_compile_long', 'soc_ga3_files_smooth_topo_old_fft_mk2_long']
exp_name_list = ['Earth_1760']
full_or_half_list=['full', 'half']
start_file=0
end_file=300
nfiles=(end_file-start_file)+1

mask_below_surface_set='-x' #Default is to mask values that lie below the surface pressure when interpolated. For some applications, e.g. Tom Clemo's / Mark Baldwin's stratosphere index, you want to have values interpolated below ground, i.e. as if the ground wasn't there. To use this option, this value should be set to '-x '. 

try:
    out_dir
except:
    out_dir = base_dir

plevs={}
var_names={}

# For 40 full levels. To convert to a list of integers without commas: ' '.join(map(str, a.astype(int))) where a = ds['phalf'].values*100 in Pa
#plevs['full']=' -p "36   142   262   463   781  1266  1973  2964  4302  6044  8238 10913 14077 17714 21784 26225 30955 35883 40910 45939 50878 55648 60185 64439 68378 71986 75261 78215 80866 83243 85379 87308 89068 90694 92220 93677 95092 96489 97884 99291"'
#plevs['half']=' -p "1     99    189    343    594    985   1569   2406   3559   5090   7052   9484  12407  15816  19683  23956  28562  33413  38412  43460  48462  53332  57997  62399  66500  70273  73712  76821  79616  82122  84369  86392  88227  89911  91479  92963  94393  95793  97185  98584 100000"'
# For 48 full levels
plevs['full']=' -p "1 3 6 11 18 27 41 62 91 132 190 270 378 524 719 976 1310 1743 2296 2998 3880 4979 6335 7998 10020 12459 15381 18857 22962 27777 33360 39683 46538 53543 60291 66509 72085 76998 81270 84945 88077 90728 92957 94823 96379 97671 98739 99611"'
plevs['half']=' -p "1 2 5 8 14 22 33 50 74 109 158 225 317 443 611 833 1125 1505 1993 2614 3399 4382 5600 7100 8931 11149 13817 17001 20775 25220 30414 36395 43064 50101 57061 63580 69482 74719 79300 83257 86644 89519 91942 93975 95673 97086 98257 99223 100000"'

var_names['full']=' rh sphum ucomp vcomp omega height temp soc_tdt_lw soc_tdt_sw soc_tdt_rad dt_tg_convection dt_tg_condensation cf reff_rad frac_liq qcl_rad rh_in_cf'
var_names['half']=' soc_flux_lw_up soc_flux_lw_down soc_flux_sw_up soc_flux_sw_down soc_flux_direct soc_flux_lw_up_clr soc_flux_sw_up_clr soc_flux_lw_down_clr soc_flux_sw_down_clr soc_flux_direct_clr'    

for exp_name in exp_name_list:
    for n in range(nfiles):
        for full_or_half in full_or_half_list:
            print(n+start_file)

            number_prefix=''

            if n+start_file < 1000:
                number_prefix='0'
            if n+start_file < 100:
                number_prefix='00'
            if n+start_file < 10:
                number_prefix = '000'

            nc_file_in = base_dir+'/'+exp_name+'/run'+number_prefix+str(n+start_file)+'/atmos_monthly.nc'
            nc_file_out = out_dir+'/'+exp_name+'/run'+number_prefix+str(n+start_file)+'/atmos_monthly_interp_'+full_or_half+'.nc'

            if not os.path.isfile(nc_file_out):
                plevel_call(nc_file_in,nc_file_out, var_names = var_names[full_or_half], p_levels = plevs[full_or_half], mask_below_surface_option=mask_below_surface_set)
         
print('execution time', time.time()-start_time)
# This will not overwrite existing files. To delete first check: 'echo run0{XXX..YYY}/atmos_monthly_interp_full.nc' then 'rm run0{XXX..YYY}/atmos_monthly_interp_full.nc'



