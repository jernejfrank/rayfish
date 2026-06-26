---------------------- Response -------------------------------------------
Dear Editor,

Thank you for considering our work for publication in PRL. Based on the reports, we perceive that both referees clearly find the paper of interest and have asked for reasonable additional material that further strengthens the manuscript.

We have addressed their concerns in the response below and in the resubmitted manuscript, most notably by adding Supplemental Material accounting for unequal brightness of point sources and experimental imperfections. We further sharpened the statement of novelty relative to previous work.

We hope you will find the revised manuscript suitable for publication in PRL.

Sincerely,
Jernej Frank,
George Brumpton,
Tommaso Tufarelli,
Gerardo Adesso,
Samanta Piano


_____________

Response to the first referee report:

We thank the referee for the positive assessment of our manuscript and for the constructive suggestions on novelty, robustness, and experimental realism. We have revised the manuscript accordingly and respond to each point below.

1. The referee asks how robust our conclusions are for more realistic cracks (unequal intensities, background light, surface roughness, aberrations, imperfect mode sorting), and in particular whether the favourable delta z^2 scaling survives beyond the toy model.

1a. We have addressed the most pressing experimental concern, imperfect mode sorting, by adding a new Supplemental Material section, "Impact of Mode-Sorting Imperfections on the Classical Fisher Information." There we model the dominant imperfection of a multi-plane light-converter sorter, namely nearest-neighbour cross-talk between adjacent Hermite-Gaussian orders, and recompute the Fisher information and Cramer-Rao bound. We find that for cross-talk up to -20 dB (which is experimentally attainable) the Fisher information is reduced but does not collapse, and mode sorting retains its advantage over direct imaging across the sub-Rayleigh regime (new figure in the Supplemental Material).

1b. Unequal/fluctuating intensities
TBD

2. The referee asks us to state more sharply what is new relative to previous work on quantum superresolution and quantum surface metrology, especially Refs. [11] and [12].

We have sharpened the statement of novelty in the Introduction. Relative to Ref. [11], which develops general expressions for the quantum Fisher information matrix in discrete quantum imaging, and Ref. [12], which treats the quantum estimation of angular and axial separations in surface metrology, the present work is new in three respects: (a) it formulates passive defect detection as a quantum hypothesis test and derives the governing Chernoff-information bounds for the quantum limit, direct imaging, and mode sorting, a task not addressed in Refs. [11,12]; (b) it treats the joint estimation of crack width and depth from the geometry of the point-spread-function manifold, rather than a single separation parameter; and (c) it provides a closed-form quantum Fisher information matrix expressed entirely through point-spread-function overlaps and their derivative couplings, which directly identifies the optimal measurement modes.

3. The referee asks how many Hermite-Gaussian modes are used and how sensitive the results are to mode truncation, mode mismatch, loss, and detector noise.

3a. The parameter information is concentrated in the lowest Hermite-Gaussian orders identified by the QFIM geometry: the (1,0) mode for the width and the (2,0) and (0,2) modes for the depth. Our numerical results therefore use a deliberately minimal mode-sorting strategy that resolves only these four modes [(0,0), (1,0), (2,0), (0,2)] together with a remainder channel collecting all unresolved light (see point 3d below). We verify insensitivity to truncation explicitly by comparing this minimal bank against progressively larger banks (box cutoffs of orders <=2 and <=5): the Cramer-Rao bounds are indistinguishable in the sub-Rayleigh regime and differ only at larger separations, where direct imaging is already preferable (new figure in the Supplemental Material).

3b. Mode mismatch and imperfect sorting are addressed by the new cross-talk analysis described in point 1 above.

3c. Detector photon loss
TBD

3d. Higher-order mode cut-off and remainder channel. We have added a Supplemental Material section, "The remainder channel," that treats mode truncation explicitly. A finite sorter resolves only a finite set of Hermite-Gaussian modes; the unresolved light is not discarded but collected in a single additional output, the dump port of a real device, which completes the measurement to a valid POVM that accounts for every photon arriving in the image plane. We now include this remainder channel in the estimation Fisher information. Because the classical Fisher information is additive over outcomes, and because merging the unresolved modes into one outcome is a coarse-graining of the fully resolved measurement, the truncated, remainder-completed, fully resolved, and quantum bounds obey J_res <= J_res+rem <= J_full <= H. We report two complementary quantities: (i) the leaked fraction 1 - sum_{j,l} P_{j,l}, i.e. the fraction of collected light not routed into a resolved mode, and (ii) the Fisher information recovered by the remainder channel. For the minimal four-mode bank, the resolved probabilities sum to unity through second order, so the leaked fraction is O(delta^4) in the sub-Rayleigh regime: the minimal bank captures essentially all photons and all parameter information. Enlarging the bank reduces the leaked fraction but improves the estimation precision only near and beyond the Rayleigh length, where direct imaging already attains the quantum limit. Truncation to a handful of modes plus a remainder channel therefore costs negligible information throughout the regime where mode sorting is advantageous (new figures in the Supplemental Material). We emphasise that this measurement-design leakage is distinct from uniform optical/detector loss (point 3c), which rescales all channel probabilities equally and does not affect the mode-sorting-versus-direct-imaging comparison.


___________

Response to the second referee report:

We thank the referee for carefully reading our manuscript and providing constructive feedback that helped us improve our paper. We have revised the manuscript and added explanations to address the questions of the referee. We hope the referee will recommend the paper for publication with these changes.

1. The referee asks whether the formalism can be extended beyond Gaussian PSFs to other PSFs such as the Airy disk, aberrated, or experimentally measured PSFs.

The framework is PSF-agnostic. The quantum Fisher information matrix, Eq.~\eqref{eq:qfim_final}, and the reduced Lyapunov equation, Eq.~\eqref{eq:L_delta_sylvester}, depend only on the Gram matrix of the PSFs and their parameter derivatives. The Gaussian assumption enters only through the closed-form overlaps given in the Supplemental Material and through the choice of a matched Hermite-Gaussian mode basis; for an Airy, aberrated, or experimentally measured PSF, the same expressions hold with the overlaps evaluated numerically. We have clarified this in the manuscript.

2. The referee asks us to clarify the meaning of axial defocus while depth information is concentrated in second-order modes.

By "axial defocus" we mean the quadratic wavefront curvature induced by an axial (z) displacement of a point source. The leading tangent direction generated by such a displacement is proportional to the second-order transverse derivative of the PSF, which for a Gaussian PSF is the combination of the (2,0) and (0,2) Hermite-Gaussian modes. This is why the depth information concentrates in second-order modes. We have clarified this point in the main text.

3. The referee asks how robust the protocol is beyond equal-intensity emitters, in particular for unknown or fluctuating intensities, and whether mode sorting remains optimal in the sub-Rayleigh regime.
TBD

4. The referee asks for the intuition behind the optimality of Hermite-Gaussian modes in the sub-Rayleigh regime.

For a Gaussian PSF, the tangent vectors generated by infinitesimal source displacements are proportional to Hermite-Gaussian functions of increasing order. A Hermite-Gaussian sorter therefore projects directly onto the information-bearing directions of the PSF manifold. In the sub-Rayleigh limit a single low order dominates each parameter (first order for the width, second order for the depth), so a small number of modes already saturates the quantum limit. We have made this geometric intuition explicit in the manuscript.

5. The referee notes that the possible future explorations are not clearly stated, and that the generality and experimental feasibility could be strengthened.

We have added an outlook paragraph to the Conclusions outlining extensions to general, aberrated, and experimentally measured PSFs, to unequal and fluctuating source intensities, and to the inclusion of background light and surface roughness.

On experimental feasibility, the new Supplemental Material section on mode-sorting cross-talk provides a first quantitative step, showing that the mode-sorting advantage survives realistic, imperfect sorting. We have additionally added a Supplemental Material section, "The remainder channel," that models the finite mode bank of a realistic sorter together with its dump-port (remainder) channel. It shows that resolving only the few information-bearing modes plus this single extra outcome -- which together account for every photon in the image plane -- suffices to reach the quantum limit in the sub-Rayleigh regime, so the protocol does not require an unbounded mode bank (see our response to the first referee, point 3d). + the TBD stuff from above that still is planned to be simulated.

We hope these amendments clarify the concerns of the referees and that they will recommend our work for publication without reservation.
