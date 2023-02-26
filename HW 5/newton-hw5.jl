# Implementation of Newton's method. 
#
# Also includes Steepest Descent algorithm (constant step size, & Armijo's rule), and 
# Momentum Descent
#
# This version: for HW3
#

using Printf
using LinearAlgebra
using Plots

# set parameters here, for all optimization algorithms
tol = 1e-6;     # tolerance on norm of gradient
MaxIter = 10000;  # maximum number of iterations of gradient descent
MaxIterNewton = 500;  # maximum number of iterations for Newton-type methods. 


# load data
include("data-hw5.jl");
z = data[:,1];
y = data[:,2];
n = length(z);   # number of data points


# sigmoidal function, and its derivatives
function s(x)
   return 1/(1+exp(-x));
end

function sp(x)
    return exp(-x)/(1+exp(-x))^2;
end

function spp(x)
    return 2*exp(-2*x)/(1+exp(-x))^3 - exp(-x)/(1+exp(-x))^2;
end


# model function: y = m(z,x), where x are parameters
function m(z,x)
    a = x[1];
    b = x[2];
    return s(a*z+b);
end

# gradient of model function
function Dm(z,x)
   a = x[1]; b = x[2]; 
   g1 = z*sp(a*z+b);
   g2 = sp(a*z+b);   ### YOU INSERT ### 
   return [g1;g2];
end

# Hessian of model function
function Dm2(z,x)
   a = x[1]; b = x[2]; 
   H = zeros(2,2);
   ### YOU INSERT ###
   H[1,1] = z^2*spp(a*z+b);
   H[1,2] = z*spp(a*z+b);
   H[2,1] = z*spp(a*z+b);
   H[2,2] = spp(a*z+b);
   ### END INSERT ###
   return H;
end


# Loss function
function F(x)
   L = 0;
   for i=1:n
      L = L + 0.5*(y[i] - m(z[i],x))^2;
   end
   return L;
end

# gradient of Loss function
function DF(x)
   g = zeros(2);
   for i=1:n
      g = g-(y[i]-m(z[i],x))*Dm(z[i],x);
   end
   return g;
end

# Hessian of Loss function
function DF2(x)
    H = zeros(2,2);
    for i=1:n
      H = H - ( (y[i]-m(z[i],x))*Dm2(z[i],x) - Dm(z[i],x)*Dm(z[i],x)');
   end
   return H;
end


function SteepestDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       Fgrad = DF(x);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          break;
       end
       # perform steepest descent step
       x = x - alpha*Fgrad;

   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return x;
end


#
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    c1 = slope, in Armijo's rule.
# output: 
#    x = final point
#
function SteepestDescentArmijo(x0, c1)

   # parameters for Armijo's rule
   alpha0 = 5.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps

   # setup for steepest descent
   x = x0;
   successflag = false;   

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;
      Fval = F(x);
      Fgrad = DF(x);

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, Fval)
         successflag = true;
         break;
      end

      # perform line search
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # print how we're doing, every 10 iterations
      #if (iter%10==0)
         #@printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
      #end
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return x;
end


# Newton's algorithm
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
# output: 
#    xsave = list of points
#
function Newton(x0)

   # setup 
   x = x0;
   successflag = false;
   xsave = zeros(length(x0),MaxIter+1);
   xsave[:,1] = x0;

   # iterate
   min_eig = 0
   for iter = 1:MaxIterNewton
      
       # compute gradient
       Fgrad = DF(x);

       @printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad));

       # check whether gradient is small enough
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("\nConverged after %d iterations, F(x) = %f\n", iter, F(x));
          println("x = ", x');
          successflag = true;
          xsave = xsave[:,1:iter];
          break;
       end

       # compute Hessian
       H = DF2(x);
       # check if hessian is positive definite
       if isnan(H[1,1]) == false && isnan(H[1,2]) == false && isnan(H[2,1]) == false && isnan(H[2,2]) == false
         if eigvals(H)[1] < min_eig || eigvals(H)[2] < min_eig
            min_eig = minimum(eigvals(H))
         end
      end
       print("current minimum eigenvalue is negative, which is", min_eig)
       ### YOU INSERT ###

       # Newton step
       x = x - inv(H)*Fgrad;    # normally you don't actually compute a matrix inverse
       
       # save point
       xsave[:,iter+1] = x;
       
   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %F\n", MaxIter, F(x))
   end
   
   return xsave;
end



# Newton's algorithm, with Hessian modification
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
# output: 
#    xsave = list of points
#
function NewtonModified(x0)

   # parameters
   backtrack = true;  # whether or not to apply backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps
   c1 = 1e-3;       # slope factor, in Armijo's rule


   # setup 
   x = x0;
   successflag = false;
   xsave = zeros(length(x0),MaxIter+1);
   xsave[:,1] = x0;

   # iterate
   min_eig = 0
   for iter = 1:MaxIterNewton
      
       # compute gradient
       Fgrad = DF(x);

       # check whether gradient is small enough
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, F(x) = %f\n", iter, F(x));
          println("x = ", x');
          successflag = true;
          xsave = xsave[:,1:iter];
          break;
       end

       # Find multiple of identity to add to Hessian, to make it positive definite
       H = DF2(x);   # Hessian
       #print("my hessian is", H)
       if isnan(H[1,1]) == false && isnan(H[1,2]) == false && isnan(H[2,1]) == false && isnan(H[2,2]) == false
         min_eig = minimum(eigvals(H))
       end
       #if eigvals(H)[1] < min_eig || eigvals(H)[2] < min_eig
        # min_eig = minimum(eigvals(H))
       #end

       tau = 1 - min_eig;

       ### YOU INSERT ### 
       B = H + tau * [1 0; 0 1] ; 
       #print("my detB is ", B[1,1]*B[2,2]-B[1,2]*B[2,1], "m B[1] is", B[1,1])  # YOU MUST POSSIBLY MODIFY B HERE
       ### END INSERT ### 
       
       d = inv(B)*Fgrad;  # descent direction
       alpha = 1;

       # find step size alpha, using backtracking
       if backtrack
          Fval = F(x);
          for k = 1:MaxBacktrack
            x_try = x - alpha*d;
            Fval_try = F(x_try);
            if (Fval_try > Fval - c1*alpha *Fgrad'*d)
               alpha = alpha * eta;
            else
               break;
            end
         end
      end

      #@printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f, tau = %6.4f, alpha = %6.4f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad),tau, alpha);

       # take step
       x = x - alpha*d;    # normally you don't actually compute a matrix inverse
       
       # save point
       xsave[:,iter+1] = x;
   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %F\n", MaxIter, F(x))
   end
   return xsave;
end

#print(Dm(1,[1,0]));
#print(Dm2(1,[1,0]));
#print(DF([1,0]))
#print(DF2([1,0]))

for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
   x = randn(2);
   println("\nMy random generated x is", x);
   NewtonModified(x)
end
#Newton(x);
#print("modified:")

#print("With alpha 0.1")
#println(SteepestDescentArmijo(x, 0.1))
#print("With alpha 0.05")
#println(SteepestDescentArmijo(x, 0.05))
#print("With alpha 0.02")
#println(SteepestDescentArmijo(x, 0.02))
#print("With alpha 0.1")
#println(SteepestDescent(x,0.1));
#print("With alpha 0.05")
#println(SteepestDescent(x,0.05));
#print("With alpha 0.02")
#println(SteepestDescent(x,0.02));
#print(x);

#print("modified:")
#NewtonModified(x)

#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point
#    alpha = step size. Constant, in this algorithm.
# output: 
#    xsave = list of points
#
function SteepestDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;
   xsave = zeros(length(x0),MaxIter+1);
   xsave[:,1] = x0;

   # perform steepest descent iterations
   for iter = 1:MaxIter

       # compute gradient
       Fgrad = DF(x);

       # print info
       @printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad));

       # check if gradient is small enough
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, F(x))
          successflag = true;
          xsave = xsave[:,1:iter];
          break;
       end

       # perform steepest descent step
       x = x - alpha*Fgrad;
       
       # save point
       xsave[:,iter+1] = x;
   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return xsave;
end



#
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point
#    c1 = slope, in Armijo's rule.
# output: 
#    xsave = list of points
#
function SteepestDescentArmijo(x0)

   # parameters
   alpha0 = 10.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps
   c1 = 1e-2;       # slope factor, in Armijo's rule

   # setup for steepest descent
   x = x0;
   successflag = false;   
   xsave = zeros(length(x0),MaxIter);
   xsave[:,1] = x0;

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;

      # compute gradient
       Fgrad = DF(x);

       # print info
       @printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad));

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, F(x))
         successflag = true;
         xsave = xsave[:,1:iter];
         break;
      end

      # perform line search
      Fval = F(x);
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # save point
      xsave[:,iter+1] = x;

      # print how we're doing, every 10 iterations
      #if (iter%5==0)
      #   @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], F(x))
      #end 
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return xsave
end








# plot contours of function
function plotcontours(xl,yl)
   xmin = -xl;
   xmax = xl;
   ymin = -yl;
   ymax = yl;
   x = xmin:0.05:xmax;
   y = ymin:0.05:ymax;
   Z = zeros(length(y), length(x));
   for i=1:length(x)
      for j=1:length(y)
        Z[j,i] = F([x[i]; y[j]]);
      end
   end
   
   contourf(x,y,Z,levels=20,aspect_ratio=:equal,fill=(true,cgrad(:inferno,rev=true)));
   #contourf(x,y,Z,levels=30,fill=(true,cgrad(:haline ,[0,0.1,1.0])));
   #contourf(x,y,Z,levels=30,fill=(true,cgrad(:deep,[0,0.2,1.0])));
   #contourf(x,y,Z,levels=30,fill=(true,cgrad(:inferno,scale=:exp)));

   # color schemes, see:
   #   http://docs.juliaplots.org/latest/generated/colorschemes/#cmocean 
   # see also
   #   https://github.com/JuliaPlots/ExamplePlots.jl/blob/master/notebooks/cgrad.ipynb
   #   https://docs.juliaplots.org/latest/generated/colorschemes/
   #   https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl
end


# plot data on top of contour plot
function plotdata(xsave,str)
   K= length(xsave[1,:]);
   println("k = ", K)
   plot!(xsave[1,1:1:K], xsave[2,1:1:K], lw = 2, marker=2, label = str,legend=:bottomright);
   #savefig("plot.png")
end


Q = [101 25;
      100 51]
print(eigmin(Q))