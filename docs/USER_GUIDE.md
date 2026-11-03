# User guide (GUI)

Start with `python3 run_app.py [material]`. The window opens paused, with RF and DNP off
unless the material says otherwise.

## Layout

**Plots:**

- **Top:** the spectrum.
- **Middle:** the signed tensor spectrum I+(R) − I−(R).
- **Bottom:** time traces of I+ and I− at the **monitor R** (solid) and at its mirror −R
  (dotted).

Tick **Show RF commands on spectrum** to overlay the scheduled and active carrier commands.
These are commands per bin, not the spread-out effect of diffusion on the hole.

**Side panel, top to bottom:**

1. **Material:** its name (hover for the description); **Load material / Save material /
   Save as…**; **Edit ALL material / search settings…** (the complete material as editable
   JSON); **RF saturation / spin diffusion…** (common RF and exchange fields with a preview);
   **Save state / Load state** (the population snapshot).
2. **Generate tensor program from CURRENT state…**: the optimizer.
3. **Run controls:** **RF ON/OFF**, **DNP ON/OFF**, spin diffusion on/off, **Pause / Run
   simulation**, **Reset populations**, time step `dt`, monitor R.
4. **RF program:** **Load program / Save program**; **Edit EVERY bin: power / start /
   duration** (a table with paste support); **Start / restart RF program**; **Stop program
   (keep evolving)**; **Pause at program end**; **Export trace / exposure CSV**.
5. **AFP:** region (optimal Q < 0 bins, or a selective L/R window), efficiency η,
   **Apply AFP now**.

## Designing an optimized burn

1. Bring the populations to the state you want to start from: reset, run DNP, or load a state.
2. Open **Generate tensor program from CURRENT state…**.
   - The materials start with no power cap. Tick **Limit total RF power** (1 = 0.1 W) and/or
     **Limit total RF exposure** (34 = 3.4 J) for experimental limits.
   - The duration range, the per-bin ceiling and **RF profile smoothing** are also set here.
3. Press **Calculate from captured state**, then **Load generated program (do not run)**. The
   installed program carries its own validated integration step.
4. Tick **Pause at program end**, press **RF ON**, then **Run simulation**. Q at the endpoint
   matches the designer's prediction.
5. Press **Run simulation** again to watch recovery afterwards.

## Continuous DNP with a steady burn

Design the burn with `tools/steady_state_rf_dnp.py` (see the README). Load its program, turn
**DNP ON**, press **RF ON** and run for several DNP time constants. To speed this up, raise
`steps_per_tick` in **Edit ALL material** (`gui` section).

## Changing physics

- **RF saturation / spin diffusion…** covers the common RF and exchange fields.
- **Edit ALL material** covers everything, including the per-packet profiles (`rf_coupling_profile`,
  `rf_width_profile`, `exchange_profile`, `dnp_profile`, `t1_profile`), `site_resolved` and
  `dipolar_reservoir`.
- Changes that alter packet capacities (grid, lineshape, site resolution) need an explicit
  reinitialization.
- A program optimized for one set of physics is not optimal for another: re-run the designer.

See [MATERIALS.md](MATERIALS.md) for every parameter.

## Timing rules

| Control | Effect |
|---|---|
| Start / restart program | Program time 0 now; RF gate on; populations unchanged |
| RF OFF | Mutes RF; program time keeps running |
| Pause simulation | Freezes everything |
| Stop program | Cancels scheduled RF; recovery continues |
| Reset populations | Initial state, t = 0; the program stays installed |

Pulse edges are integrated exactly. The display timer only sets the redraw rate.

## Files

- **Material:** `.json` (`ssrf.material.v1`).
- **State:** `.npz` snapshot, fingerprinted. It includes site-resolved populations and β when
  those are active.
- **Program:** `.json` or `.csv` (bin, power, start, duration).
- **Designer export:** a folder with the program, report, predicted endpoint and starting state.
