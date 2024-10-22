# EE125 Project 1 - Part 1, Complex Numbers
# Sept. 16, 2024

import numpy as np

# 1. Define a complex variable z as z = -1 - 2j
z = complex('-1-2j')

# 2. Find the real and imaginary part of z.
z_real = z.real
z_im   = z.imag
print(f"For z, z_real = {z.real}, z_im = {z_im}")

# 3. Determine the magnitude of z using both ABS and
#    the formula involving R and Im. parts of z. Equal?
z_mag = np.abs(z)
z_mag_formula = np.sqrt(np.pow(z_real, 2) + np.pow(z_im, 2))
print(f"The magnitude derived from np.abs and formula's being equal is: {z_mag == z_mag_formula}")

# 4. Determine phase of z using both arctan and ANGLE function.
z_phase = np.angle(z) # in radians
z_phase_arctan = np.arctan(z_im / z_real)

print(f"The phase of z from np.phase: {z_phase}, which is in radians")
print(f"The phase of z from arctan: {z_phase_arctan}, in radians")
# these values differ, as arctan spits the value out regardless of quadrant, while np.phase is always correct quad

# 5. Calculate (z + z*) / 2 and (z - z*) / 2j
print(f"(z+z*) / 2 is: {(z + z.conjugate()) / 2}")
print(f"(z-z*) / 2 is {(z - z.conjugate()) / 2}")
# these relate to the components of z, where the first is z_real, the second is z_im
