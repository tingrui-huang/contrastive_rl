# Pre-training derivation: a scalar diagonal-error budget

Fix x,z=x_prime independently uniform on [-1,1], c=0, bank {-3}, and the
full 1-Lipschitz bound in x at each fixed z. Both losses use the same z marginal.
Integrating separately on [-1,z] and [z,1] gives E[|x-z||z]=(1+z^2)/2.

Projection of a feasible response onto [-3,infinity) preserves its full bound,
weakly improves bank distance, and improves the diagonal square if d=G(z,z)<-3.
Thus restrict to d>=-3. At fixed d, the closest feasible response to the bank is
G_d(x,z)=max(-3,d-|x-z|), simultaneously for all x. This is feasible itself.
The minimized conditional objective is J_z(d)=d^2+lambda*E[(3+d-|x-z|)_+^2],
strictly convex in d. Write a=lambda/(1+lambda). Its unsaturated stationary
point is d*(z)=-a*(5-z^2)/2. For 0<=lambda<1 and t=|z|<=1, its minimum margin
above the bank is 2-t-a*(5-t^2)/2, decreasing in t, hence at least
1-2*a=(1-lambda)/(1+lambda)>0. The stationary point is in the valid branch
everywhere and strict convexity proves global optimality, not just stationarity.
For lambda>0 the optimal response is G*=d*(z)-|x-z|. At lambda=0 only the
diagonal is identified; -|x-z| is merely an evaluation reference for the control.

E[(3-E[D|z])^2]=82/15 and E[(3-D)^2]=17/3. Therefore
diagonal MSE=a^2*82/15, failure MSE=17/3+(a^2-2*a)*82/15, and
J*=lambda*17/3-lambda^2/(1+lambda)*82/15.
The maximum absolute optimal diagonal is 2.5*a.

Predetermine epsilon=0.01 and lambda=0.004 before any new training.
Then a=1/251, maximum |d*|=0.0099601593625498, diagonal RMSE=0.009315101151,
failure MSE=5.623194340831, J*=0.022579548472776, and minimum margin above
the bank=0.99203187251. The optimal budget slack is only 0.00003984063745.
This is an illustrative native scalar outcome budget, not a PointMaze tolerance.
A trained model can exceed it through approximation, finite-data or optimization
error. Finite probes alone do not certify a continuum bound.

The conditioner is piecewise affine in z, whereas d* is quadratic. An independent
same-architecture reference interpolates d* with 33 knots, h=1/16, and has exact
relative action slopes +1/-1. Its maximum diagonal approximation error is
a*h^2/8. Since J_z(d)-J_z(d*)=(1+lambda)*(d-d*)^2 in this regime, integration
gives gap (1+lambda)*a^2*h^4/120. This tightly brackets the finite architecture's
optimum above J*. It is evaluation only; never used to initialize or train.

The new run changes only lambda, reusing the unchanged training routine, all
three original initializations, datasets, minibatch streams, optimizer and 2500
steps. Old lambda=0.1 joint and lambda=0 diagonal-only checkpoints are replayed
read-only. Each objective gap uses its own weight; raw differently weighted
totals are not compared as performance scores. No settings or final checkpoint
are selected using the already-inspected evaluation sets.
