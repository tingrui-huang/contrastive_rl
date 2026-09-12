# Pre-training reference: fixed scalar failure bank

Fix the bank {-3}, lambda=0.1, L=1, and z=x_prime uniformly distributed on
[-1,1] in BOTH losses. Draw x conditionally as Uniform[-1,1], independently
of z; do not redraw z during any x handling. Lower outcomes are worse only
by the convention of this toy task. No pointwise next-state labels are given.

For a fixed diagonal d=G(z,z), projection of any response onto [-3,infinity)
preserves 1-Lipschitzness, weakly decreases bank distance and weakly decreases
the diagonal square if d<-3. We can restrict attention to d>=-3. For fixed d,
the best response toward -3 under the full action bound is
G_d(x,z)=max(-3,d-|x-z|). It is feasible for every x and attains the nearest
bank value allowed by the diagonal constraint, simultaneously for all x.

Let D=|x-z|. The conditional objective is
J_z(d)=d^2+lambda*E[(3+d-D)_+^2]. It is strictly convex in d.
For uniform conditional x, m(z)=E[D|z]=(1+z^2)/2. On the unsaturated branch,
d*(z)=-lambda/(1+lambda)*(3-m(z))=-(5-z^2)/22.
Checking the branch globally: D<=1+|z| and
3+d*(z)-(1+|z|)>=9/11>0. Thus no relevant optimal output reaches the bank.
This is the unique diagonal optimum; G*=d*(z)-D is the associated full-bound
response. Its minimum is -24/11, strictly above -3. The optimum diagonal ranges
from -5/22 to -2/11; a zero diagonal is NOT optimal for this two-loss objective.
The quadratic bank loss does not give the -lambda/2 shift of a linear objective.

Integrating over z yields diagonal MSE=82/1815, diagonal RMSE=0.2125536717,
failure MSE=8563/1815, and J*=853/1650=0.51696969697.
The shift-only family G=d(z) has optimum d=-3/11 and total 9/11.
The best zero-anchor response -D has total 17/30. These are analytic evaluation
references, not additional trained arms and not inputs to optimization.

The existing finite ReLU conditioner is piecewise affine in z, so its exact
diagonal cannot equal this quadratic everywhere. The global feasible-class
optimum is therefore a lower bound, not a claim of exact membership in this NN.
An independently constructed model of exactly the same architecture linearly
interpolates d* at 33 knots on [-1,1], with the correct bounded action slopes.
Its maximum diagonal approximation error is h^2/88 for h=1/16. Because the
conditional objective difference is 1.1*(d-d*)^2, its population objective gap
is exactly 1.1*h^4/(22^2*30)=1.155968868e-9. Thus the optimum of the actual
architecture lies between J* and J*+1.16e-9. This reference is used ONLY in
evaluation, never as an initializer, loss target, checkpoint or candidate selector.

For d=G(z,z) and h=G(x,z)-d, decompose improvement from the zero response in
a stated order: 9-E[(3+d)^2] from moving the diagonal first, followed by
E[(3+d)^2]-E[(3+d+h)^2] from action-dependent change. These add exactly to the
bank-distance reduction, but are order-dependent because of the cross term.
Also report recentered bank distance E[(3+h)^2]; recentering is a diagnostic,
never a replacement for the final emitted model.
