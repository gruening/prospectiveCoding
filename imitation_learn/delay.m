function xdel = delay(x,n)  % Adds n colums of zeros to the left of the matrix x and delets the last n columns
   xdel0=[zeros(size(x,1),n), x]; xdel=xdel0(:,1:size(x,2));