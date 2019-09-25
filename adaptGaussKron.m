function I = adaptGaussKron(func,a,b,tol,varargin)
% adaptGaussKron  Uses the 7 point Gauss rule with the 15 point Kronrod 
%                 rule to evaluate the integral of a function from a to b. 
%
% I = adaptGaussKron(fun,a,b,tol)
%
% func = the function whose integral is being approximated
% a = lower limit of the integral
% b = upper limit of the integral
% tol = tolerance of the approximation. The adaptGaussKron function returns
%       an approximation when abs(I1-I2) < tol*I1, where I1 is the
%       approximation of the first integral, taken with a large step, and
%       I2 is the approximation of a second integral, taken with a 
%       smaller step. 
%
% I = approximate value of the integral of the function f, from a to b

% Evaluate integral once before passing it on
I1 = G7K15(func,a,b,varargin{:});

I = doAdaptGaussKron(func,a,b,tol,I1,varargin{:});

function I = doAdaptGaussKron(func,a,b,tol,I1,varargin)
% doAdaptGaussKron  Generates an approximation by taking smaller steps
%                   until the specified tolerance is met.
%
% I = doAdaptGaussKron(func,a,b,tol,I1)
%
% Same parameters as adaptGaussKron plus:
% I1 = current value of the integral

c = a + (b-a)/2;
% Evaluate to the left and right of c
I2l = G7K15(func,a,c,varargin{:});
I2r = G7K15(func,c,b,varargin{:});

% Sum the approximations over the two intervals
I2 = I2l + I2r;  

% Check if tolerance is met
if abs(I2-I1) < tol*abs(I1)   
  % Tolerance met, return approximation 
  I = I2;  
  return;          
else  
  % Tolerance not met, take a smaller step
  I =   doAdaptGaussKron(func,a,c,tol,I2l,varargin{:}) ...      
      + doAdaptGaussKron(func,c,b,tol,I2r,varargin{:});        
end

function [I,absE] = G7K15(func,a,b,varargin)
% G7K15  Gauss-Kronrod quadrature pair of order 7 and 15
%
%  I = G7K15(f,a,b)
% [I,absE] = G7K15(f,a,b)
%
% func = func whose integral is going to be approximated
% a = lower limit of the integral
% b = upper limit of the integral
%
% I = approximation to integral of f from a to b. Computed by using the 
%     15 point Kronrod rule and adding that to the result from the 7 point
%     Gauss rule.
% abserr = estimate of absolute error. Computed as the difference between
%          the 7 point Gauss and the 15 point Kronrod rules

% 7 point Gaussian weights
GaussW = [ 0.129484966168869693270611432679082;   
           0.279705391489276667901467771423780;
           0.381830050505118944950369775488975;
           0.417959183673469387755102040816327 ];

% Kronrod nodes, where Gaussian nodes are at Nodes(2:2:8)
Nodes = [ 0.991455371120812639206854697526329; 
          0.949107912342758524526189684047851;  
          0.864864423359769072789712788640926;  
          0.741531185599394439863864773280788;
          0.586087235467691130294144838258730;
          0.405845151377397166906606412076961;
          0.207784955007898467600689403773245;
          0                                   ];

% 15 point Kronrod weights
KronW  =[ 0.022935322010529224963732008058970;  
          0.063092092629978553290700663189204;
          0.104790010322250183839876322541518;
          0.140653259715525918745189590510238;
          0.169004726639267902826583426598550;
          0.190350578064785409913256402421014;
          0.204432940075298892414161999234649;
          0.209482141084727828012999174891714 ] ;
    
% Find midpoint of the interval
mid = (b-a)/2;  
% Shift nodes into the interval (a,b)
shiftedN = (a+b)/2 + mid*[-Nodes(1:8); Nodes(7:-1:1)];  
% Evaluate the function at the 15 nodes
f = feval(func,shiftedN,varargin{:});                      

% 7 pt Gauss
gSum = mid*(sum(GaussW(1:3).*f(2:2:6)) + sum(GaussW(4:-1:1).*f(8:2:14))); 
% 15 pt Kronrod
I = mid*(sum(KronW(1:8).*f(1:8)) + sum(KronW(7:-1:1).*f(9:15)));  

if nargout==2 
    % Compute absolute error if desired
    absE = abs(I-gSum);  
end