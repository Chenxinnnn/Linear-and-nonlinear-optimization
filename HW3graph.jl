
using Pkg, JuMP, GLPK, Plots, GLM
Pkg.add("JuMP")
Pkg.add("Ipopt")
Pkg.add("Plots")
Pkg.add("GLM")


x = [1,2,2.5,3,3,4,5];
g = [2,1,3,3,5.5,3.5,5];
scatter(x,g)
display(plot!((x) -> 0.80769 + 0.84615 * x,0,5,label="h"))
