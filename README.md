# Sample commands can be found in main.py
# Example simulation results can be found in 210818_vortex_code_example.pdf
This script calculates the z-component of the magnetic field of a superconducting Pearl vortex in a thin film at a certain height, then convolves with a SQUID pickup loop. Here, I assumed a default penetration depth of 500um. The script can be easily adapted for a bulk Abrikosov vortex by using the proper scalar potential expression given by Kogan. 
The function getVortex is adapted from a Matlab code by John R. Kirtley. PhysRevLett.92.157006 shows why the exact Fourier term is chosen as well as shows experimental vortex images.

