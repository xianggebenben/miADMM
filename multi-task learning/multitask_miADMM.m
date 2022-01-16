function [w,r_history,s_history,obj_history]=multitask_miADMM(X,Y,alpha,lambda,isfix)
n=length(X);
p=size(X{1},2);
rho=1000;
z1=zeros(p,n);
rng('default');
w=2*rand(p,n)-1;
u1=zeros(p,n);
max_iter=15000;
threshold=1e-5;
for i=1:max_iter
    i
    z1_old=z1;
    if ~isfix
        lambda =lambda+10;
    end
    % update w
    for j=1:n
        if(j==1)
            [w(:,j)]=update_first_last(X{j},Y{j},alpha,lambda,rho,z1(:,j),u1(:,j),w(:,j+1),p,w(:,j));
        elseif (j==n)
            [w(:,j)]=update_first_last(X{j},Y{j},alpha,lambda,rho,z1(:,j),u1(:,j),w(:,j-1),p,w(:,j));
        else
            [w(:,j)]=update_w(X{j},Y{j},alpha,lambda,rho,z1(:,j),u1(:,j),w(:,j-1),w(:,j+1),p,w(:,j));
        end
    end
    % update z1
    z1=update_z1(w,rho,u1,alpha);
    r=w-z1;
    s=rho*(z1_old-z1);
    u1=u1+rho*r;
    r=norm(r);
    s=norm([reshape(s,size(s,1)*size(s,2),1)]);
    r_history(i)=r;
    s_history(i)=s;
    obj=objective(alpha,X,Y,w,u1,z1,rho,n);
    obj_history(i)=obj;
    if(s<threshold) && (r<threshold) 
        break;
    end
end
end
function obj=objective(alpha,X,Y,w,u1,z1,rho,n)
obj=0;
for i=1:n
    obj=obj+norm(Y{i}-X{i}*w(:,i))^2/length(Y)+alpha*norm(z1(:,i))^2;
end
obj=obj+sum(sum(u1.*(w-z1)))+rho/2*norm(w-z1,'fro')^2;
end
function w=update_w(X,y,alpha,lambda,rho,z1,u1,w_last,w_next,p,w)
n=length(y);
w=inv(2*X'*X/n+2*alpha*eye(p)+rho*eye(p))*(2*X'*(y)/n+rho*z1-u1);
if lambda>=1e5
w(w_last.*w_next<0)=0;
index=find(w_last>0&w_next>0);
index=[index;find(w_last>0&w_next==0)];
index=[index;find(w_last==0&w_next>0)];
w(index)=max(w(index),0);
index=find(w_last<0&w_next<0);
index=[index;find(w_last<0&w_next==0)];
index=[index;find(w_last==0&w_next<0)];
w(index)=min(w(index),0);
end
end
function w=update_first_last(X,y,alpha,lambda,rho,z1,u1,w_neighbor,p,w)
w=inv(2*X'*X+2*alpha*eye(p)+rho*eye(p))*(2*X'*y+rho*z1-u1);
if lambda>=1e5
index=find(w_neighbor>0);
w(index)=max(w(index),0);
index=find(w_neighbor<0);
w(index)=min(w(index),0);
end
end
function z1=update_z1(w,rho,u,alpha)
z1=(rho*w+u)/(2*alpha+rho);
end
