"""MaterialTuning-8 extended dynamics: site-resolved packets and a dipolar reservoir.

State: populations n[s, i, level] for sites s (1, or 2 = C-D, O-D when
site_resolved), frequency bins i and levels (+, 0, -), plus the dipolar inverse
temperature beta (1/R units). Packet (s, i) has its I+ transition at offset R_i
and its I- transition at -R_i; its mirror partner is (s, N-1-i).

Energy bookkeeping (offset units, Larmor part removed): level + of packet i
lies R_i below 0, level - lies R_i above... more precisely the I+ (+<->0) and
I- (0<->-) transitions have offsets R_i and -R_i, so the spin offset energy is
E_spin = -sum R_i (n+ + n-). The reservoir energy is E_d = -C_d beta.

Every process that changes E_spin by dE exchanges -dE with the reservoir and
obeys detailed balance with respect to beta to first order (high-temperature
dipolar reservoir; the Zeeman populations themselves are NOT linearized):
  RF carrier j on transition with offset w:  J = W [ d - beta (w - R_j) s ]
  same-branch flip-flop i<->j:               J = H [ (..) + beta E/2 (..) ]
  cross-branch flip-flop:                    E = R_i + R_j
  DQ exchange:                               energy neutral (no beta term)
Exchange-only dynamics conserve E_spin + E_d exactly (tested).

With site_resolved=False and dipolar off this engine is not used (the legacy
code runs). With one site and beta=0 it reproduces the legacy equations.
Requires spectral RF, orientation_corr_fraction=0, kernel_cutoff_widths=0.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

from .lineshape import level_populations_from_PQ

P_, Z_, M_ = 0, 1, 2


class ExtendedCore:
    def __init__(self, model):
        from .spectral_rf import RFRateOperator
        p = model.params
        self.p = p
        if p.rf_response_mode != 'spectral':
            raise ValueError('Extended dynamics require spectral RF')
        if p.orientation_corr_fraction != 0 or p.kernel_cutoff_widths != 0:
            raise ValueError('Extended dynamics require no orientation correlation and no kernel cutoff')
        R = model.Rplus.astype(float); N = len(R); self.R = R; self.N = N
        self.site_resolved = bool(p.site_resolved)
        if self.site_resolved:
            if p.line_model != 'multisite_butanol':
                raise ValueError('site_resolved requires multisite_butanol')
            from .multisite_lineshape import site_components
            cd, od = site_components(R, epsilon=1, cd_scale=p.line_cd_scale_R, od_ratio=p.line_od_scale_ratio,
                hwhm=p.line_gamma, eta_cd=p.line_eta_cd, eta_od=p.line_eta_od,
                od_fraction=p.line_od_fraction, points=p.line_phi_points)
            a = np.vstack([np.maximum(-cd.imag, 0.), np.maximum(-od.imag, 0.)])
            tot = a.sum(axis=0)
            share = np.divide(a, tot, out=np.full_like(a, .5), where=tot > 0)
            self.mu = share*model.mu[None, :]          # site capacities; sum over sites = model.mu
            w_od, c2_cd, c2_od = model.packet_site_geometry()
            self.c2 = np.vstack([c2_cd, c2_od])
            self.S = 2
        else:
            self.mu = model.mu[None, :].copy(); self.S = 1
        S = self.S
        # ---- per-site mechanism profiles ----
        def site_profile(prefix):
            if not self.site_resolved:
                return model.mechanism_profile(prefix)[None, :]
            a2 = float(getattr(p, prefix+'_theta_p2')); a4 = float(getattr(p, prefix+'_theta_p4'))
            od = float(getattr(p, prefix+'_od_factor'))
            dens = model._density_power_profile(float(getattr(p, prefix+'_density_exponent', 0.0)),
                                                float(getattr(p, prefix+'_density_clip', 12.0)))
            c2 = self.c2
            ang = 1. + a2*(1.5*c2-.5) + a4*(35.*c2*c2-30.*c2+3.)/8.
            f = ang*np.array([1., od])[:, None]*dens[None, :]
            if np.any(f <= 0) or not np.all(np.isfinite(f)):
                raise ValueError(f'{prefix} site profile must be positive')
            return f
        band = model.rf_band_response() if self.site_resolved else model.rf_packet_response()
        self.c = (site_profile('rf_coupling')*band[None, :]) if self.site_resolved else band[None, :]
        gamma = float(p.rf_homogeneous_hwhm_R)*site_profile('rf_width')
        self.ops = []
        for s in range(S):
            g = gamma[s]
            hw = float(g[0]) if np.all(g == g[0]) else g
            self.ops.append(RFRateOperator(R, np.ones(N), strength=p.rf_spectral_strength, hwhm=hw,
                                           packet_response=np.ones(N)))
        self.sx = np.sqrt(site_profile('exchange'))[:, :, None]          # (S,N,1)
        self.dnp_w = model.capacity_rate_weights()[None, :]*site_profile('dnp')
        self.t1 = float(p.t1_rate)*site_profile('t1')
        x = float(p.exchange_intersite_factor)
        self.M = np.array([[1.]]) if S == 1 else np.array([[1., x], [x, 1.]])
        # ---- exchange kernel (finite Toeplitz via FFT, as legacy) ----
        col = model._spectral_overlap(np.arange(N)*model.dR)
        self.L = next_fast_len(2*N-1)
        embed = np.zeros(self.L); embed[:N] = col; embed[-N+1:] = col[1:][::-1]
        self.kfft = rfft(embed)[None, :, None]
        self.S2 = .5*model._spectral_overlap(2*R)[None, :, None]
        self.Rb = R[None, :, None]                                          # (1,N,1)
        # ---- reservoir ----
        self.dipolar = bool(p.dipolar_enabled)
        self.Cd = float(p.dipolar_capacity_ratio)*float(np.sum(self.mu*R[None, :]**2))
        self.r1d = float(p.dipolar_relaxation_rate)
        self.beta_L = float(p.dipolar_beta_lattice)
        self.xi = float(p.dipolar_exchange_coupling)
        self._targets = {}

    # ---------------- helpers ----------------
    def split(self, n_total, P_like=None):
        """Distribute bin totals over sites with equal fractions (initialization / resync)."""
        frac = n_total/np.maximum(n_total.sum(axis=1, keepdims=True), 1e-300)
        return self.mu[:, :, None]*frac[None, :, :]

    def collapse(self, n):
        return n.sum(axis=0)

    def target(self, P):
        key = float(P)
        if key not in self._targets:
            pref = level_populations_from_PQ(max(-.999999, min(.999999, key)), None)
            self._targets[key] = self.mu[:, :, None]*pref[None, None, :]
        return self._targets[key]

    def _conv(self, V):
        return irfft(self.kfft*rfft(V, n=self.L, axis=1), n=self.L, axis=1)[:, :self.N]

    def _mix(self, V):
        return np.einsum('st,tnc->snc', self.M, V)

    def Ho(self, V):
        return self.sx*self._mix(self._conv(self.sx*V))

    def Xo(self, V):
        W = self.sx*V
        return self.sx*self._mix(self._conv(W[:, ::-1])-self.S2*W)

    def Co(self, V):
        W = (self.sx*V).sum(axis=1, keepdims=True)
        return self.sx*self._mix(np.broadcast_to(W, V.shape).copy())

    def _rates(self, u):
        """Per-site RF fields: G (rate), Hh (first offset moment), Q (second moment)."""
        R = self.R; out = []
        Ru = R*u; R2u = R*R*u
        for s, op in enumerate(self.ops):
            c = self.c[s]
            kp, kpR, kpR2 = op.Kp(u), op.Kp(Ru), op.Kp(R2u)
            km, kmR, kmR2 = op.Km(u), op.Km(Ru), op.Km(R2u)
            Gp = c*kp; Hp = c*(R*kp-kpR); Qp = c*(R*R*kp-2*R*kpR+kpR2)
            Gm = c*km; Hm = c*(-R*km-kmR); Qm = c*(R*R*km+2*R*kmR+kmR2)
            out.append((Gp, Hp, Qp, Gm, Hm, Qm))
        return out

    def _flags(self, dnp_on):
        p = self.p
        scale = (p.diffusion_scale*(p.microwave_diffusion_factor if dnp_on else 1.)
                 if p.diffusion_enabled else 0.)
        decay = self.t1.copy()
        source = self.t1[:, :, None]*self.target(p.t1_p_eq)
        if dnp_on and p.dnp_rate:
            d = p.dnp_rate*self.dnp_w
            decay = decay+d
            source = source+d[:, :, None]*self.target(p.p_dnp_sat)
        return scale, decay, source

    # ---------------- dynamics ----------------
    def rhs(self, n, beta, u, dnp_on):
        """Return (dn, dbeta, E_released). n: (S,N,3); u: (N,) carrier powers."""
        scale, decay, source = self._flags(dnp_on)
        b = float(beta) if self.dipolar else 0.
        bx = b*self.xi
        R = self.Rb
        dn = source-decay[:, :, None]*n
        E = 0.
        np_, n0, nm = n[..., P_], n[..., Z_], n[..., M_]
        Rr = self.R[None, :]
        if scale:
            kx = scale*self.p.cross_branch_ratio; kdq = scale*self.p.double_quantum_ratio
            A = self.Ho(n)
            B = self.Ho(R*n) if bx else None
            fp = np_*0; fm = np_*0
            fp = n0*A[..., P_]-np_*A[..., Z_]
            fm = nm*A[..., Z_]-n0*A[..., M_]
            if bx:
                fp = fp+.5*bx*(Rr*(n0*A[..., P_]+np_*A[..., Z_])-(n0*B[..., P_]+np_*B[..., Z_]))
                fm = fm+.5*bx*((nm*B[..., Z_]+n0*B[..., M_])-Rr*(nm*A[..., Z_]+n0*A[..., M_]))
            fp *= scale; fm *= scale
            dn[..., P_] += fp; dn[..., Z_] += fm-fp; dn[..., M_] -= fm
            E += float(np.sum(Rr*(fp-fm)))
            if kx:
                AX = self.Xo(n)
                cp = n0*AX[..., Z_]-np_*AX[..., M_]
                cm = n0*AX[..., Z_]-nm*AX[..., P_]
                if bx:
                    BX = self.Xo(R*n)
                    cp = cp+.5*bx*(Rr*(n0*AX[..., Z_]+np_*AX[..., M_])+(n0*BX[..., Z_]+np_*BX[..., M_]))
                    cm = cm+.5*bx*(Rr*(n0*AX[..., Z_]+nm*AX[..., P_])+(n0*BX[..., Z_]+nm*BX[..., P_]))
                cp *= kx; cm *= kx
                dn[..., P_] += cp; dn[..., M_] += cm; dn[..., Z_] -= cp+cm
                E += float(np.sum(Rr*(cp+cm)))
            if kdq:
                AC = self.Co(n)
                z = kdq*(nm*AC[..., P_]-np_*AC[..., M_])
                dn[..., P_] += z; dn[..., M_] -= z
        if np.any(u):
            for s, (Gp, Hp, Qp, Gm, Hm, Qm) in enumerate(self._rates(u)):
                dp = np_[s]-n0[s]; sp = .5*(np_[s]+n0[s])
                dm = n0[s]-nm[s]; sm = .5*(n0[s]+nm[s])
                Jp = Gp*dp-b*Hp*sp; Jm = Gm*dm-b*Hm*sm
                dn[s, :, P_] -= Jp; dn[s, :, Z_] += Jp-Jm; dn[s, :, M_] += Jm
                E -= float(np.sum(Hp*dp-b*Qp*sp+Hm*dm-b*Qm*sm))
        dbeta = (-E/self.Cd-self.r1d*(b-self.beta_L)) if self.dipolar else 0.
        return dn, dbeta, E

    def vjp(self, n, beta, u, a, ab, dnp_on):
        """Return (g_n, g_beta, g_u): gradients of sum(a*dn) + ab*dbeta."""
        scale, decay, source = self._flags(dnp_on)
        dip = self.dipolar
        b = float(beta) if dip else 0.
        bx = b*self.xi
        R = self.Rb; Rr = self.R[None, :]
        w = -ab/self.Cd if dip else 0.                     # weight on E_released
        g = -decay[:, :, None]*a
        gb = (-self.r1d*ab) if dip else 0.
        np_, n0, nm = n[..., P_], n[..., Z_], n[..., M_]
        aP, aZ, aM = a[..., P_], a[..., Z_], a[..., M_]
        gu = np.zeros(self.N)
        if scale:
            kx = scale*self.p.cross_branch_ratio; kdq = scale*self.p.double_quantum_ratio
            A = self.Ho(n); gA = np.zeros_like(n)
            B = self.Ho(R*n) if bx else None; gB = np.zeros_like(n)
            psi = scale*(aP-aZ+w*Rr)          # SQ+ : dn+ += fp, dn0 -= fp, E += R fp
            chi = scale*(aZ-aM-w*Rr)          # SQ- : dn0 += fm, dn- -= fm, E -= R fm
            hp = .5*bx*Rr
            g[..., Z_] += psi*(A[..., P_]*(1+hp))
            g[..., P_] += psi*(A[..., Z_]*(-1+hp))
            gA[..., P_] += psi*n0*(1+hp); gA[..., Z_] += psi*np_*(-1+hp)
            g[..., M_] += chi*(A[..., Z_]*(1-hp))
            g[..., Z_] += chi*(A[..., M_]*(-1-hp))
            gA[..., Z_] += chi*nm*(1-hp); gA[..., M_] += chi*n0*(-1-hp)
            if bx:
                g[..., Z_] += psi*(-.5*bx*B[..., P_]); g[..., P_] += psi*(-.5*bx*B[..., Z_])
                gB[..., P_] += psi*(-.5*bx*n0); gB[..., Z_] += psi*(-.5*bx*np_)
                g[..., M_] += chi*(.5*bx*B[..., Z_]); g[..., Z_] += chi*(.5*bx*B[..., M_])
                gB[..., Z_] += chi*(.5*bx*nm); gB[..., M_] += chi*(.5*bx*n0)
                gb += self.xi*float(np.sum(psi*.5*(Rr*(n0*A[..., P_]+np_*A[..., Z_])-(n0*B[..., P_]+np_*B[..., Z_]))))
                gb += self.xi*float(np.sum(chi*.5*((nm*B[..., Z_]+n0*B[..., M_])-Rr*(nm*A[..., Z_]+n0*A[..., M_]))))
            elif dip and self.xi:
                # beta = 0 exactly: B still enters the beta-derivative.
                B0 = self.Ho(R*n)
                gb += self.xi*float(np.sum(psi*.5*(Rr*(n0*A[..., P_]+np_*A[..., Z_])-(n0*B0[..., P_]+np_*B0[..., Z_]))))
                gb += self.xi*float(np.sum(chi*.5*((nm*B0[..., Z_]+n0*B0[..., M_])-Rr*(nm*A[..., Z_]+n0*A[..., M_]))))
            g += self.Ho(gA)
            if bx: g += R*self.Ho(gB)
            if kx:
                AX = self.Xo(n); gAX = np.zeros_like(n); gBX = np.zeros_like(n)
                s3 = kx*(aP-aZ+w*Rr); s4 = kx*(aM-aZ+w*Rr)
                g[..., Z_] += (s3+s4)*AX[..., Z_]*(1+hp)
                g[..., P_] += s3*AX[..., M_]*(-1+hp)
                g[..., M_] += s4*AX[..., P_]*(-1+hp)
                gAX[..., Z_] += (s3+s4)*n0*(1+hp)
                gAX[..., M_] += s3*np_*(-1+hp)
                gAX[..., P_] += s4*nm*(-1+hp)
                need_BX = bx or (dip and self.xi)
                BX = self.Xo(R*n) if need_BX else None
                if bx:
                    g[..., Z_] += (s3+s4)*.5*bx*BX[..., Z_]
                    g[..., P_] += s3*.5*bx*BX[..., M_]
                    g[..., M_] += s4*.5*bx*BX[..., P_]
                    gBX[..., Z_] += (s3+s4)*.5*bx*n0
                    gBX[..., M_] += s3*.5*bx*np_
                    gBX[..., P_] += s4*.5*bx*nm
                if need_BX:
                    gb += self.xi*float(np.sum(s3*.5*(Rr*(n0*AX[..., Z_]+np_*AX[..., M_])+(n0*BX[..., Z_]+np_*BX[..., M_]))))
                    gb += self.xi*float(np.sum(s4*.5*(Rr*(n0*AX[..., Z_]+nm*AX[..., P_])+(n0*BX[..., Z_]+nm*BX[..., P_]))))
                g += self.Xo(gAX)
                if bx: g += R*self.Xo(gBX)
            if kdq:
                AC = self.Co(n); gAC = np.zeros_like(n)
                tau = kdq*(aP-aM)
                g[..., M_] += tau*AC[..., P_]; g[..., P_] -= tau*AC[..., M_]
                gAC[..., P_] += tau*nm; gAC[..., M_] -= tau*np_
                g += self.Co(gAC)
        if np.any(u) or True:
            R1 = self.R
            for s, (Gp, Hp, Qp, Gm, Hm, Qm) in enumerate(self._rates(u)):
                dp = np_[s]-n0[s]; sp = .5*(np_[s]+n0[s])
                dm = n0[s]-nm[s]; sm = .5*(n0[s]+nm[s])
                al_p = aZ[s]-aP[s]; al_m = aM[s]-aZ[s]
                g_dp = al_p*Gp-w*Hp; g_sp = -b*al_p*Hp+w*b*Qp
                g_dm = al_m*Gm-w*Hm; g_sm = -b*al_m*Hm+w*b*Qm
                g[s, :, P_] += g_dp+.5*g_sp
                g[s, :, Z_] += -g_dp+.5*g_sp+g_dm+.5*g_sm
                g[s, :, M_] += -g_dm+.5*g_sm
                if dip:
                    gb += float(np.sum(-al_p*Hp*sp-al_m*Hm*sm+w*(Qp*sp+Qm*sm)))
                # command gradient: coefficients on G, Hh, Q
                c = self.c[s]; op = self.ops[s]
                cG = c*al_p*dp; cH = c*(-b*al_p*sp-w*dp); cQ = c*(w*b*sp)
                gu += op.KpT(cG)
                gu += op.KpT(cH*R1)-R1*op.KpT(cH)
                gu += op.KpT(cQ*R1*R1)-2*R1*op.KpT(cQ*R1)+R1*R1*op.KpT(cQ)
                cG = c*al_m*dm; cH = c*(-b*al_m*sm-w*dm); cQ = c*(w*b*sm)
                gu += op.KmT(cG)
                gu += -op.KmT(cH*R1)-R1*op.KmT(cH)
                gu += op.KmT(cQ*R1*R1)+2*R1*op.KmT(cQ*R1)+R1*R1*op.KmT(cQ)
        return g, gb, gu

    def outgoing_bound(self, u, beta, dnp_on):
        """Conservative per-packet outgoing-rate bound for Euler step selection."""
        scale, decay, _ = self._flags(dnp_on)
        b = abs(float(beta)) if self.dipolar else 0.
        Rmax = float(np.max(np.abs(self.R)))
        out = decay.copy()
        for s, (Gp, Hp, Qp, Gm, Hm, Qm) in enumerate(self._rates(u)):
            out[s] += np.abs(Gp)+np.abs(Gm)+b*(np.abs(Hp)+np.abs(Hm))
        if scale:
            mu = self.mu[:, :, None]
            f = 1.+b*self.xi*Rmax
            out += 2*scale*f*self.Ho(mu)[..., 0]
            if self.p.cross_branch_ratio: out += 2*scale*self.p.cross_branch_ratio*f*np.abs(self.Xo(mu))[..., 0]
            if self.p.double_quantum_ratio: out += scale*self.p.double_quantum_ratio*self.Co(mu)[..., 0]
        return float(out.max())

    def spin_offset_energy(self, n):
        return float(-np.sum(self.R[None, :]*(n[..., P_]+n[..., M_])))
