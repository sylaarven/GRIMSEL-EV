
print('READING config_local filei {}'.format(__file__))
import os
FN_XLSX = os.path.abspath('/Users/arthurrinaldi/Dropbox/paper_1/data/model_data/input_levels_190105.xlsx')
DATABASE = 'grimsel_1' 
SCHEMA = 'lp_input_example'

PSQL_USER = 'postgres'
PSQL_PASSWORD = 'postgres'
PSQL_PORT = 5432
PSQL_HOST = 'localhost'

# PATH_CSV = os.path.abspath('/Users/sylaarv1/Dropbox/grimsel_arven/input_data')

# BASE_DIR = os.path.abspath('/Users/sylaarv1/Dropbox/grimsel_arven/data')

# (for the pc - a ma posht the path osht per server)
#PATH_CSV = os.path.abspath('C:/Users/sylaarv1/Desktop/Work/grimsel_switch_drive_ev_HPC/grimsel_arven_ev/input_data') 
PATH_CSV = os.path.abspath('/home/users/s/syla/scratch/grimsel_dsr_ee_dhw_bass_ev/input_data')
# (for the pc - a ma posht the path osht per server)
#BASE_DIR = os.path.abspath('C:/Users/sylaarv1/Desktop/Work/grimsel_switch_drive_ev_HPC/grimsel_arven_ev/build_input/build_input_files')
BASE_DIR = os.path.abspath('/home/users/s/syla/scratch/grimsel_switch_drive_ev/grimsel_arven_ev/build_input/build_input_files')
