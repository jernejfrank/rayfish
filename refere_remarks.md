
----------------------------------------------------------------------
Report of Referee A -- LQ20641/Frank
----------------------------------------------------------------------

This manuscript develops a quantum statistical framework for passive optical surface metrology and defect detection. The authors model a surface as an incoherent ensemble of point emitters and use quantum Fisher information and quantum Chernoff bounds to derive ultimate limits for estimating and detecting sub-diffraction features. As an application, they study a minimal three-point-source crack model and compare direct imaging with Hermite-Gaussian spatial mode sorting.

The paper is clearly written and addresses a timely topic at the interface of quantum imaging, optical metrology, and defect detection. For shallow cracks, direct imaging gives a Chernoff exponent scaling as xi_DI ~ delta z^4 / 72, whereas mode sorting reaches the quantum-limited scaling xi_MS ~ delta z^2 / 12. This is a clean and physically meaningful result, showing that suitable spatial-mode measurements can access defect information largely hidden from conventional intensity imaging.

I find the work potentially suitable for Physical Review Letters, but I recommend revision before publication. The main issue is that the concrete crack model is highly idealized: three equal-intensity incoherent point sources represent the defect. The authors should clarify how robust the main conclusions are for more realistic cracks, unequal intensities, background light, surface roughness, aberrations, or imperfect mode sorting. In particular, it would be useful to know whether the favorable δz^2 scaling is expected to survive beyond the toy model.

The authors should also state more sharply what is new relative to previous work on quantum superresolution and quantum surface metrology, especially Refs. [11] and [12]. As written, the novelty appears to lie mainly in the passive defect-detection formulation and the Chernoff-bound analysis, but this should be made explicit.

Finally, the experimental assumptions should be clarified. How many Hermite-Gaussian modes are used in the numerical results? How sensitive are the results to mode truncation, mode mismatch, loss, and detector noise? Since the proposed advantage relies on weak signals in higher-order modes, these practical issues are important.

The manuscript contains a clear and interesting quantum-limited imaging result, but the authors should strengthen the discussion of novelty, robustness, and experimental realism before publication in Physical Review Letters.


The manuscript studies the quantum limits of passive optical surface metrology and defect
detection via tools from quantum parameter estimation and quantum hypothesis testing,
respectively. The authors model a surface as an incoherent ensemble of point emitters imaged
through a diffraction-limited optical system and derive fundamental bounds for estimating
geometrical surface parameters and discriminating between defective or defect-free structures.
The framework is formulated using a non-orthogonal basis representation of the PSF, described
by Gaussian beam, allowing the QFIM to be expressed directly in terms of PSF overlaps and
their derivative structure.
As a representative application, the paper considers a minimal surface-crack model consisting
of three incoherent point sources of equal intensity, parameterized by crack width and depth.
The authors compare conventional DI with Hermite-Gaussian spatial MS and show that the
latter approaches the ultimate quantum limits for simultaneous estimation of crack width and
depth in the sub-Rayleigh regime. In particular, they identify that the dominant information about
transverse crack width is carried by first-order Hermite-Gaussian modes, while depth
information is concentrated in second-order modes associated with axial defocus.
The work further extends the analysis to binary hypothesis testing for crack detection. The
authors derive the quantum Chernoff bound and compare it with the Chernoff information
achievable via DI and MS. A principal result is that MS achieves the optimal quadratic scaling of
the Chernoff exponent with crack depth, whereas direct imaging exhibits only quartic scaling,
implying significantly poorer distinguishability of shallow defects.
I have the following questions:
1. The authors have assumed Gaussian PSFs. Can the formalism be extended beyond
Gaussian PSFs to other PSFs such as Airy Disk, aberrated or other experimentally
measured PSFs?
2. Can you please clarify the meaning of axial defocus while depth information is
concentrated in second-order modes?
3. Beyond equal-intensity emitters, how robust is the protocol? What happens if the point
sources have unknown or fluctuating intensities? Does MS remain optimal for the
sub-Rayleigh regime?
4. What are the intuitions behind the optimality of Hermite-Gaussian modes in the
sub-Rayleigh regime?
Although the results are very interesting, the possible future explorations are not clearly written
in the manuscript. While the present work is an excellent addition to the optical surface
metrology, I feel that the results remain somewhat limited in generality, and the experimental
feasibility of the proposed framework is not yet sufficiently established to justify publication in
Phys Rev Lett
