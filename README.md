# Wald space

With this package you can compute geodesics and Fréchet means and many more things
in wald space and (hopefully) in the future also in BHV space and other spaces that 
define a geometry on the space of trees/forests.

The most important classes are the classes `Wald` and `WaldSpace`, Wald is representing a 
tree or forest in many different characterisations, and WaldSpace is the go-to class
for creating other objects like geodesics or Fréchet means.

You load create an instance via
`ws = WaldSpace(geometry='wald')`
and then compute geodesics between different walds by calling `ws.g.w_path(p=p, q=q, alg='variational')`,
where `p` and `q` are instances of class `Wald`.

There are also small examples already provided.
Call `python3 examples/cobwebs/spiderweb1.py` to calculate the firing of geodesics in different directions and get 
plots of the results in the same directory.