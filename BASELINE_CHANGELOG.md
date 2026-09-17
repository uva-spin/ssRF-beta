# Change log

## Mirror-balance diagnostic revision

- Preserved the Voigt RF population operator and all prior diffusion physics.
- Added an independent **DIFFUSION ON/OFF** GUI control.
- Added live RF, diffusion, and net rate readouts for both mirror components.
- Changed the uncalibrated demonstration default from `K0=50` to `K0=5`, so the
  RF-created mirror feature is not immediately hidden by recovery.
- Added regression tests for the diffusion toggle and the mirror source/sink
  decomposition.
