using JuMP, GLPK

# using GLPK solver
model = Model(GLPK.Optimizer) 

# define variables
@variable(model, x >= 0)
# equivalent way:
# @variable(model, x)
# @constraint(model, x >= 0)
@variable(model, y >= 0)
@variable(model, z >= 0)

# @variable(model, lb <= x <= ub, start=x0) # define varaibles with lower/upper bounds and initial value

# add constraints
@constraint(model, x - 2y <= 0)
@constraint(model, y + z <= 1)


# define objective function
@objective(model, Max, x - y -z)

# print model
print(model)

# To solve the optimization problem, we call the optimize function.
#! means to modify original value
optimize!(model)

# show status after optimizing -- is the problem infeasible or unbounded?
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)

# show the optimal values (multiple ways to do this)
println("optimal x = ", value(x))   # method 1: your own print statement with value() command
@show value(x);                     # method 2: using @show with built-in printing
y_opt = JuMP.value(y);
println("optimal y = ", y_opt);     # method 3: save optimal value in a new variable
# @show value(y);
@show objective_value(model);