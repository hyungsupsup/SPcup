import glob, os
import shutil

merge_file = glob.glob(os.path.join('C:/Users/USER/Desktop/Cifar10/airplane/', '.*jpg'))
merge_file = merge_file[0:6]
file_name = os.listdir(merge_file)

os.makedirs('C:/Users/USER/Desktop/Cifar10_test6000/airplane/')
for i, val in enumerate(merge_file) :
    shutil.copyfile(val, 'C:/Users/USER/Desktop/Cifar10_test6000/airplane/'+file_name[i])
