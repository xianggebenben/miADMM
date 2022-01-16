function [MSE,MSLE,MAE,EV,R2]=multitask_test(W,X,Y)
n=size(W,2);
m=0;
y=[];
y_hat=[];
for i=1:n
    m=m+length(Y{i});
    y=[y;Y{i}];
    y_hat=[y_hat;X{i}*W(:,i)];
end
MSE=norm(y-y_hat)^2/m;
MSLE=norm(log(1+y)-log(1+y_hat))^2/m;
MAE=norm(y-y_hat,1)/m;
EV=1-var(y-y_hat)/var(y);
R2=1-norm(y-y_hat)^2/norm(y-mean(y))^2;
end