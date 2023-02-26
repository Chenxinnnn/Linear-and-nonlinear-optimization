
using JuMP, Ipopt

###############################################
# Version 1: unconstrained
###############################################

model = Model(Ipopt.Optimizer) # different optimizer than for LPs
# set_silent(model)              # suppress the verbose printout when optimizing

# define variables with initial guesses
@variable(model, x1, start = 0.0)
@variable(model, x2, start = 0.0)
@variable(model, x3, start = 0.0)
@variable(model, x4, start = 0.0)
@variable(model, x5, start = 0.0)
@variable(model, x6, start = 0.0)
@variable(model, x7, start = 0.0)
@variable(model, x8)
@variable(model, x9)

@constraint(model, x1 - x8 - x9 >= -2)
@constraint(model, x1 + x8 + x9 >= 2)
@constraint(model, x2 - x8 - 2x9 >= -1)
@constraint(model, x2 + x8 + 2x9 >= 1)
@constraint(model, x3 - x8 - 2.5x9 >= -3)
@constraint(model, x3 + x8 + 2.5x9 >= 3)
@constraint(model, x4 - x8 - 3x9 >= -3)
@constraint(model, x4 + x8 + 3x9 >= 3)
@constraint(model, x5 - x8 - 3x9 >= -100)
@constraint(model, x5 + x8 + 3x9 >= 100)
@constraint(model, x6 - x8 - 4x9 >= -3.5)
@constraint(model, x6 + x8 + 4x9 >= 3.5)
@constraint(model, x7 - x8 - 5x9 >= -5)
@constraint(model, x7 + x8 + 5x9 >= 5)

@NLobjective(model, Min, x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2 + x7^2)
# @objective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2) # wrong use; need leading "NL"

###############################################
# Version 2: linear constraint
###############################################
# @constraint(model, x + y == 10)

###############################################
# Version 3: nonlinear constraint
###############################################
# @NLconstraint(model, x^2+y^4 >= 5)

###############################################
# Version 4: quadratic constraint
###############################################
# @NLconstraint(model, x^2+y^2 >= 5) 
# @constraint(model, x^2+y^2 >= 5) # you can use @constraint for quadratic constraints

optimize!(model)
println("a = ", value(x8), " b = ", value(x9))
@show objective_value(model)

using JuMP, GLPK, LinearAlgebra

vector_model = Model(GLPK.Optimizer)

# constraint matrix coefficients
A = [1 0 0 0 0 0 0 -1 -1;
     1 0 0 0 0 0 0 1 1;
     0 1 0 0 0 0 0 -1 -2;
     0 1 0 0 0 0 0 1 2;
     0 0 1 0 0 0 0 -1 -2.5;
     0 0 1 0 0 0 0 1 2.5;
     0 0 0 1 0 0 0 -1 -3;
     0 0 0 1 0 0 0 1 3;
     0 0 0 0 1 0 0 -1 -3;
     0 0 0 0 1 0 0 1 3;
     0 0 0 0 0 1 0 -1 -4;
     0 0 0 0 0 1 0 1 4;
     0 0 0 0 0 0 1 -1 -5;
     0 0 0 0 0 0 1 1 5]

# constraint right-hand side
b = [-2; 2; -1; 1; -3; 3; -3; 3; -100; 100; -3.5; 3.5; -5; 5]

# objective function coefficients
c = [1; 1; 1; 1; 1; 1; 1; 0; 0]

# vectorized variable: @variable(vector_model, x[1:n])
@variable(vector_model, x[1:9])
@constraint(vector_model, x[1:7].>= 0)
# equivalent way:
# @variable(vector_model, x[1:4])
# @constraint(vector_model, x .>= 0)

# vectorized constraints
# . dot means element wise
@constraint(vector_model, A * x .>= b) # define through Linear algebra
# Or elementwise:
# n = length(b);
# @constraint(vector_model, [i=1:n],sum(A[i,j]*x[j] for j=1:4) <= b[i])
# The following (without the ".") causes an error:
# @constraint(vector_model, A * x <= b)


@objective(vector_model, Min, c' * x)
# ‘ means transpose
# equivalent way if you are "using LinearAlgebra"
# @objective(vector_model, Max, dot(c, x))

print(vector_model)

optimize!(vector_model)

@show value.(x);
# @show value(x); # This is the wrong format; it will cause an error
# x_opt = JuMP.value.(x);
# println("value(x)= ", x_opt);
@show objective_value(vector_model);




using JuMP, GLPK, LinearAlgebra

vector_model = Model(GLPK.Optimizer)


# constraint matrix coefficients
A = [1 5;
     1 8]

b = [4;6] 
c = [9;60]
@variable(vector_model, y[1:2] >= 0)
@constraint(vector_model, A * y .>= b)
@objective(vector_model, Min, c' * y)
print(vector_model)
optimize!(vector_model)
@show value.(y);
@show objective_value(vector_model);




using Pkg, JuMP, GLPK, Plots, GLM
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("Plots")
Pkg.add("GLM")


x = [1,2,2.5,3,3,4,5];
g = [2,1,3,3,5.5,3.5,5];
scatter(x,g)
plot!((x) -> 0.80769 + 0.84615 * x,0,5,label="h")