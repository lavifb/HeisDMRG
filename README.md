DMRG Code
====

Uses Intel MKL and PRIMME libraries.

Thanks to http://troydhanson.github.io/uthash/ for easy to use hashtable.

Code is sped up considerably using the PRIMME eigensolver. It can be compiled without PRIMME but this is not recommended as the performance gains are substantial.
PRIMME can be compiled from https://github.com/primme/primme and is needed for faster eigenvalue solving.

Todo
----

- Fleshed out runtime tester and progress bar

- Time dependent code
	- Trotter evolution?

- Write to HD for very large matrices

- Sparse storage of matrices

####Maybe

- Variable m for set truncation error

- Use Input File for parameters
	- Full Hamiltonian


-----

Written by Lavi Blumberg <lavi@stanford.edu>
