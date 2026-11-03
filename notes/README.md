# Notes

Technical notes on the physics and the software. Each note is a single LaTeX file that compiles
as is (pdfLaTeX, or Overleaf), with the compiled PDF next to it.

| Note | Contents |
|---|---|
| [ssRF simulation package: physics and implementation](ssrf_simulation_note.pdf) ([.tex](ssrf_simulation_note.tex)) | How the simulator works: spin-1 levels and the Pake lineshape, packet populations and mirror bookkeeping, the RF operator, the three spin-1 flip-flop channels and their stationary state, DNP, the dipolar reservoir, AFP, time integration, the tensor optimizer, the calibrated materials, data generation, the learning models, the GUI, tests, calibration, and the exact settings used in the present simulations. |
| [Spin-1 population dynamics in polarized deuteron targets](spin1_rate_equations.pdf) ([.tex](spin1_rate_equations.tex)) | The rate equations for spin 1 with all mechanisms (RF, AFP, relaxation, spectral spin diffusion, dipolar reservoir, DNP), with the solid-state literature. Not specific to the software. |
