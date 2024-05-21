# GP_EOS_generator
Gaussian Process emulator with physics constraints to determine an EOS in different dimensions

This implements the following physics constraints in 1D:
$$ \frac{\partial P}{\partial T}>0\quad \text{(2nd law of thermodynamics)} $$
$$ \frac{\partial^2 P}{\partial T^2}>0\quad\text{(compressibility)} $$
$$ c_s^2 \ge 0 \;\text{and}\; c_s^2 < 1\quad\text{(causality)} $$

