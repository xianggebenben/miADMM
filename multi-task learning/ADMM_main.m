clear;
load school.mat
p=size(X{1},2);
n=length(Y);
for i=1:n
m=length(Y{i});
rng('default');
seq=unidrnd(m,1,round(m*0.5));
Xtrain{i}=X{i}(seq,:);
Ytrain{i}=Y{i}(seq,:);
Xtest{i}=X{i};
Ytest{i}=Y{i};
Xtest{i}(seq,:)=[];
Ytest{i}(seq,:)=[];
end
optimal_miADMM_alpha=0.01;
optimal_miADMM_lambda=10e5;
[W_miADMM,r_history,s_history,obj_history]=multitask_miADMM(Xtrain,Ytrain,optimal_miADMM_alpha,optimal_miADMM_lambda,true);
[MSE_miADMM,MSLE_miADMM,MAE_miADMM,EV_miADMM,R2_miADMM]=multitask_test(W_miADMM,Xtest,Ytest);
save('constant_ADMM_result.mat')