%% spcup 2022

% 파일 불러오기 
addpath 'C:\Users\USER\Desktop\spcup_2022\spcup_2022_training_part1'; % 폴더 경로 추가
path = 'C:\Users\USER\Desktop\spcup_2022\spcup_2022_training_part1\*.wav'; % wav 파일 경로
List = dir(path); % wav 파일 리스트 불러오기
L = length(List);

labels = readtable('C:\Users\USER\Desktop\labels.csv'); % 라벨 파일 불러오기
fn = table2array(labels(:,1)); % 파일명
label = table2array(labels(:,2)); % 라벨

label_list = [0 1 2 3 4];
cnt0 = 0; cnt1 = 0; cnt2 = 0; cnt3 = 0; cnt4 = 0;
for i = 1:L
    i
    
    [data, fs] = audioread(List(i).name);
    if label(i)==label_list(1)
        cnt0 = cnt0+1;
        eval(sprintf('data0.data0_%04d = data;', cnt0));
        label0(cnt0) = label(i);
        fs0(cnt0) = fs;
    elseif label(i)==label_list(2)
        cnt1 = cnt1+1;
        eval(sprintf('data1.data1_%04d = data;', cnt1));
        label1(cnt1) = label(i);
        fs1(cnt1) = fs;
    elseif label(i)==label_list(3)
        cnt2 = cnt2+1;
        eval(sprintf('data2.data2_%04d = data;', cnt2));
        label2(cnt2) = label(i);
        fs2(cnt2) = fs;
    elseif label(i)==label_list(4)
        cnt3 = cnt3+1;
        eval(sprintf('data3.data3_%04d = data;', cnt3));
        label3(cnt3) = label(i);
        fs3(cnt3) = fs;
    elseif label(i)==label_list(5)
        cnt4 = cnt4+1;
        eval(sprintf('data4.data4_%04d = data;', cnt4));
        label4(cnt4) = label(i);
        fs4(cnt4) = fs;
    end
    
end

%% training
data0_0001 = getfield(data0, 'data0_0001')