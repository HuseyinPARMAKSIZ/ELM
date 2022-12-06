All codes are in the testing phase yet!!!


%% Original  Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG - NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE %%
%% Improved and edited by me last quarter of 2022 (ELM for Radio Frequency Fingerprint Classification Problems.)

clear all; close all; clc; warning off
%------------------Loading datasets ----------------% 
%-----Load Ready-RF dataset -----------------------------%
load('Ready6Ftrvete.mat'); % load('Ready9Ftrvete.mat');
Yd_train=train_data(:,1)'; X=train_data(:,2:size(train_data,2))'; X=zscore(X);
Yd_test=test_data(:,1)';  Xt=test_data(:,2:size(test_data,2))'; Xt=zscore(Xt);
%-----Definitions--------%
Elm_Type = 1; REGRESSION=0; CLASSIFIER=1; ite=1; C=max(eig(X'*X)); % regularization factor                              
N=size(X,2); %training data size
Nt=size(Xt,2); %testing data size
Nin=size(X,1);% input neurons size
Nhn=100; % hidden neurons size
ActivationFunction='tansig';
DenSay=25; % iteration size

if Elm_Type~=REGRESSION
    %----- Preprocessing the data of classification
    sorted_target=sort(cat(2,Yd_train,Yd_test),2); % 1*3985 bütün sınıflar sıralı yan yana 
    label=zeros(1,1);  % Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(N+Nt)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    Non=number_class; %output neurons size, class size
       
    %----- Preprocessing the targets of training
    temp_T=zeros(Non, N);
    for i = 1:N
        for j = 1:number_class
            if label(1,j) == Yd_train(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    Yd_train=temp_T*2-1; % 4*2378 hangi sınıfa aitse 1 diğerleri -1

    %----- Preprocessing the targets of testing
    temp_TV_T=zeros(Non, Nt);
    for i = 1:Nt
        for j = 1:number_class
            if label(1,j) == Yd_test(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    Yd_test=temp_TV_T*2-1;  % 4*1019 hangi sınıfa aitse 1 diğerleri -1 

end    %   end if of Elm_Type

while ite<=DenSay
%-- Calculate weights & biases
start_time_train=tic; %cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases Bhn (b_i) of hidden neurons
InputWeight=rand(Nhn,Nin)*2-1;
Bhn=rand(Nhn,1);
tempH=InputWeight*X;
ind=ones(1,N);
BiasMatrix=Bhn(:,ind);    %   Extend the bias matrix Bhn to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'logsig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'tansig'}
        %%%%%%%% Radial basis function
        H = tansig(tempH);               
end
clear tempH;  %   Release the temporary hidden neuron output matrix H
%--- Calculate output weights OutputWeight (beta_i)
%OutputWeight=pinv(H') * T';      
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 
OutputWeight=(eye(size(H,1))/C+H * H') \ H * Yd_train';      % faster method 2
TrainigTime=toc(start_time_train);
timeTr(ite)=TrainigTime;

%   Y is the actual output of the training data
Y=(H' * OutputWeight)';                            
clear H;

%--- Calculate the output of testing input
start_time_test=tic;
tempH_test=InputWeight*Xt;
ind=ones(1,Nt);
%--- Extend the bias matrix Bhn to match the demention of H
BiasMatrix=Bhn(:,ind); 
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
   case {'logsig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'tansig'}
        %%%%%%%% Sigmoid 
        H_test = tansig(tempH_test);   
end
%--- TY is the actual output of the testing data
TY=(H_test' * OutputWeight)';  
TestingTime=toc(start_time_test);
timeTe(ite)=TestingTime;

 %---  Calculate training & testing classification accuracy
if Elm_Type == CLASSIFIER
WrongGuessClass=0;    WrongGuessClassT=0;
Yd_tr=Yd_train; Yd_te=Yd_test;
%--- Calculate training classification accuracy
    for i = 1 : size(Yd_tr, 2)
        [x, i_Labeld]=max(Yd_tr(:,i));
        [x, i_labela]=max(Y(:,i));
            if i_labela~=i_Labeld
                WrongGuessClass=WrongGuessClass+1;
            end
    end
    TrainingAccuracy(ite)=1-WrongGuessClass/size(Yd_tr,2);
%--- Calculate testing classification accuracy
    for i = 1 : size(Yd_te, 2)
        [x, i_Labeld]=max(Yd_te(:,i));
        [x, i_labela]=max(TY(:,i)); % TY: the actual output of the testing data
            if i_labela~=i_Labeld
                WrongGuessClassT=WrongGuessClassT+1;
            end
    end
    TestingAccuracy(ite)=1-WrongGuessClassT/size(Yd_te,2);
end % Classifier_if
ite=ite+1; %iteration increment
end % while ite

TrAcc=mean(TrainingAccuracy)
TrTime=mean(timeTr)
TeAcc=mean(TestingAccuracy)
TeTime=mean(timeTe)
