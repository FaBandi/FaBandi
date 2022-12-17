function [ TestingAccuracy] = CalculateTestAccuracy(Weight,Elm_Type, train_data, test_data)
 global  NumberofHiddenNeurons

 

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset

T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset

TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break;
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
    
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break;
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
    
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
% start_time_train=cputime;


InputWeight=reshape(Weight(1:NumberofInputNeurons*NumberofHiddenNeurons),[NumberofInputNeurons NumberofHiddenNeurons]);
BiasofHiddenNeurons=Weight(NumberofInputNeurons*NumberofHiddenNeurons+1:end)';

tempH=InputWeight'*P;
clear P;                                            %   Release input of training data
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
%%%%%%%% Sigmoid
H = 1 ./ (1 + exp(-tempH));

clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper


clear H;
%%%%%%%%%%% Calculate the output of testing input
% start_time_test=cputime;
tempH_test=InputWeight'*TV.P;
clear TV.P;             %   Release input of testing data
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;

%%%%%%%% Sigmoid
H_test = 1 ./ (1 + exp(-tempH_test));

TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data

if Elm_Type == CLASSIFIER
    %%%%%%%%%% Calculate training & testing classification accuracy

    MissClassificationRate_Testing=0;
    

    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end