%% add required libraries to the path
% clear; close all; clc;
%addpath(genpath('C:\Users\TCLAB-JG\Desktop\extract_lfcc.m'))

%% Prepare Data
dataDir = 'C:\Users\TCLAB-JG\Desktop\spcup_2022\audiofile_shorted500';
ads = audioDatastore(dataDir,'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

%% Labeling
ads.Labels(1:100,1) = '0';
ads.Labels(101:200,1) = '1';
ads.Labels(201:300,1) = '2';
ads.Labels(301:400,1) = '3';
ads.Labels(401:500,1) = '4';

%% train test split
[adsTrain, adsTest] = splitEachLabel(ads,0.8);

adsTrain
trainDatastoreCount = countEachLabel(adsTrain)

adsTest
testDatastoreCount = countEachLabel(adsTest)

%% Feature Extraction


[sampleTrain, dsInfo] = read(adsTrain);
reset(adsTrain)

fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);

features = [];
labels = [];
while hasdata(adsTrain)
    
    [audioIn,dsInfo] = read(adsTrain);
    
    melC = mfcc(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    feat = [melC,f0];
    
    voicedSpeech = isVoicedSpeech(audioIn,fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    label = repelem(dsInfo.Label,size(feat,1));
    
    features = [features;feat];
    labels = [labels,label];
end

%정규화
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;

%% Training a Classifier
trainedClassifier = fitcknn( ...
    features, ...
    labels, ...
    'Distance','euclidean', ...
    'NumNeighbors',5, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(labels));

k = 5;
group = labels;
c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);

validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun','ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

validationPredictions = kfoldPredict(partitionedModel);
figure
cm = confusionchart(labels,validationPredictions,'title','Validation Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';



%% Testing the Classifier
features = [];
labels = [];
numVectorsPerFile = [];
while hasdata(adsTest)
    [audioIn,dsInfo] = read(adsTest);
    
    melC = mfcc(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
    feat = [melC,f0];
    
    voicedSpeech = isVoicedSpeech(audioIn,fs,windowLength,overlapLength);
    
    feat(~voicedSpeech,:) = [];
    numVec = size(feat,1);
    
    label = repelem(dsInfo.Label,numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;feat];
    labels = [labels,label];
end
features = (features-M)./S;

prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));

figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(labels,prediction,'title','Test Accuracy (Per Frame)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure('Units','normalized','Position',[0.4 0.4 0.4 0.4])
cm = confusionchart(adsTest.Labels,r2,'title','Test Accuracy (Per File)');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

function voicedSpeech = isVoicedSpeech(x,fs,windowLength,overlapLength)

pwrThreshold = -40;
[segments,~] = buffer(x,windowLength,overlapLength,'nodelay');
pwr = pow2db(var(segments));
isSpeech = (pwr > pwrThreshold);

zcrThreshold = 1000;
zeroLoc = (x==0);
crossedZero = logical([0;diff(sign(x))]);
crossedZero(zeroLoc) = false;
[crossedZeroBuffered,~] = buffer(crossedZero,windowLength,overlapLength,'nodelay');
zcr = (sum(crossedZeroBuffered,1)*fs)/(2*windowLength);
isVoiced = (zcr < zcrThreshold);

voicedSpeech = isSpeech & isVoiced;

end
