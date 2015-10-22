
% Forward-Inverse sequence learning, from Richard Hahnloser, received Nov
% 13, 2010

nsong=200;
nu=200;
ny=200;

nlearn=5000;
nlearn2=5000;


U=randn(nsong,nu)/sqrt(nu); % syrinx
ud=randn(nu,1);

Y=randn(ny,nsong)/sqrt(ny); % auditory pathway
tut_song=U*ud;
yd=Y*tut_song;
%yd=.2*ones(ny,1);


m=randn(nu,ny)/sqrt(nu);
f=randn(ny,nu);

eta=0.001;
e=zeros(1,nlearn);
%% learn FM
for i=1:nlearn
    u=randn(nu,1);
    y=Y*U*u;
    dy=(y-f*u);
    f=f+eta*dy*u';
    e(i)=dy'*dy;
end
figure(1);clf;plot(e); ylabel('Forward error');

%% learn IM
eta2=0.001;
scale=1;
e2=zeros(1,nlearn2);
for i=1:nlearn
    y=randn(ny,1);
    u=m*y;
    u=randn(nu,1);
    y2=f*u;
    u2=m*y2;
    du=(scale*u-u2);
    m=m+eta2*du*y2';
    e2(i)=du'*du;
end

figure(2);clf;plot(e2); ylabel('Inverse error');

u=m*yd/scale;
ypred=Y*U*u;
figure(3);clf; plot(ypred,'k');hold on;plot(yd,'r');
legend('ypred','yd');
[r,p]=corrcoef(yd,ypred);
fprintf('Corr. coeff: %.4f\n',r(1,2));
