%% spcup 2022 train data 정리 (label별 sort)

clear
clc

addpath 'C:\Users\USER\Desktop\spcup_2022\spcup_2022_training_part1';
path = 'C:\Users\USER\Desktop\spcup_2022\spcup_2022_training_part1\*.wav';
List = dir(path);
L = length(List);

labels = readtable('labels_sort.csv');
fn = table2array(labels(:,1));
label = table2array(labels(:,2));

label_list = [0 1 2 3 4];
cnt0 = 0; cnt1 = 0; cnt2 = 0; cnt3 = 0; cnt4 = 0;
for i = 1:L
    i
    
    [data, fs] = audioread(cell2mat(fn(i)));
    if label(i)==label_list(1)
        cnt0 = cnt0+1;
        eval(sprintf('data0.data0_%04d = data;', cnt0));
        label0(cnt0) = label(i);
        fs0(cnt0) = fs;
        filename = sprintf('%d_%d.wav', label_list(1), cnt0);
        audiowrite(filename, data, fs);
    elseif label(i)==label_list(2)
        cnt1 = cnt1+1;
        eval(sprintf('data1.data1_%04d = data;', cnt1));
        label1(cnt1) = label(i);
        fs1(cnt1) = fs;
        filename = sprintf('%d_%d.wav', label_list(2), cnt1);
        audiowrite(filename, data, fs);
    elseif label(i)==label_list(3)
        cnt2 = cnt2+1;
        eval(sprintf('data2.data2_%04d = data;', cnt2));
        label2(cnt2) = label(i);
        fs2(cnt2) = fs;
        filename = sprintf('%d_%d.wav', label_list(3), cnt2);
        audiowrite(filename, data, fs);
    elseif label(i)==label_list(4)
        cnt3 = cnt3+1;
        eval(sprintf('data3.data3_%04d = data;', cnt3));
        label3(cnt3) = label(i);
        fs3(cnt3) = fs;
        filename = sprintf('%d_%d.wav', label_list(4), cnt3);
        audiowrite(filename, data, fs);
    elseif label(i)==label_list(5)
        cnt4 = cnt4+1;
        eval(sprintf('data4.data4_%04d = data;', cnt4));
        label4(cnt4) = label(i);
        fs4(cnt4) = fs;
        filename = sprintf('%d_%d.wav', label_list(5), cnt4);
        audiowrite(filename, data, fs);
    end
    
end

%% fft plot

filenumber = 0;
for i = label_list
    filenumber = filenumber+1
    
    eval(sprintf('tmp_data = struct2cell(data%d);', i));
    L = length(tmp_data);
    for j = 1:L
        data = cell2mat(tmp_data(j));
        Y = fft(data);
        sample_num = length(data);
        P2 = abs(Y/sample_num);
        P1 = P2(1:sample_num/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        eval(sprintf('f = fs%d(%d)*(0:(sample_num/2))/sample_num;', i, j));
        
        figure()
        plot(f, P1)
        eval(sprintf("title('Single-Sided Amplitude Spectrum of data%d_%d th', 'Interpreter', 'none')", i, j));
        xlabel('f (Hz)')
        ylabel('|P1(f)|')
        filename = sprintf('data%d_%dth.png', i, j);
        saveas(gcf, filename)
        close all
    end
    
end
