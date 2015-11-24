function xcircshift = delayperm(x,n)  % Circles the columns of matrix x to the right by n columns. 
   xcircshift=circshift(x',n)';