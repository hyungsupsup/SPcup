

#src = 'C:/Users/TCLAB-JG/Desktop/spcup_2022/audiofile'
#dir = 'C:/Users/TCLAB-JG/Desktop/spcup_2022/audiofile_shorted500'

import shutil
import os

source = 'C:/Users/TCLAB-JG/Desktop/spcup_2022/audiofile - 복사본'
dest1 = 'C:/Users/TCLAB-JG/Desktop/spcup_2022/audiofile_shorted500'


f_name=None
for n in range(0,5):
    for f_n in range(1,101):
        f_name=str(n)+'_'+str(f_n)+'.wav'
        shutil.move(source + '/' + f_name, dest1)

# for f in files:
#     if f.startswith('0_'):
#         for i in range(0,100):
#             shutil.move(source+'/'+files[i],dest1)
#     if f.startswith('1_'):
#         for j in range(0, 100):
#             shutil.move(source + '/' + files[j], dest1)