clc;
clear;
close all;
data=xlsread('bupa');
train=data(1:199,:);
test=data(200:end,:);

[ TrainingAccuracy, TestingAccuracy,Sensitivity,Specificity,Gmean,Fscore] = elm(train, test, 1, 300, 'sig')