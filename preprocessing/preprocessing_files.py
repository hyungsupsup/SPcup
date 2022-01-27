import os,glob,random,shutil
from pathlib import Path
dir_path=Path('F:/건강관리를 위한 음식 이미지/data4training') #abc 폴더 경로
parent_folder=os.path.join(str(dir_path.parent)+"/Validation") #abc의 부모폴더 +validation


for dir in os.listdir(dir_path):
    if not dir.endswith("json"):
        path=os.path.join(str(dir_path)+'/'+dir) #abc폴더내 이미지 폴더주소 순회
        file_list = os.listdir(path)
        json_folder=os.path.join(parent_folder+'/'+dir+' json') #validation폴더내 json 폴더 경로
        final_path = os.path.join(path + " json")
        os.makedirs(final_path) #abc 폴더내 새로운 json폴더 만들어줌
        for file in file_list:
            file=file[:-4]

            try:
                shutil.copy(json_folder+"/"+file+".json", final_path)

            except FileNotFoundError:
                pass
    else:
        pass

