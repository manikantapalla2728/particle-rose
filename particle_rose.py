import numpy as np, cv2, sys, subprocess, time, os, multiprocessing as mp

cv2.setNumThreads(2)
W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = 2*np.pi
rng = np.random.default_rng(11)

def sm(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(x*(x*6 - 15) + 10)

def nz(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

def mix(a, b, m):
    return a*(1 - m[:, None]) + b*m[:, None]

# Visual Palette
C_DARK = np.array([0.42, 0.0, 0.025])
C_MID  = np.array([0.96, 0.035, 0.065])
C_HI   = np.array([1.0, 0.16, 0.15])
GOLD   = np.array([1.0, 0.75, 0.28])
PINK_SOFT = np.array([0.98, 0.40, 0.48])
GREEN_D = np.array([0.015, 0.26, 0.07])
GREEN_L = np.array([0.10, 0.65, 0.20])

bag = {k: [] for k in ['pos', 'nrm', 'col', 'cls', 't0', 'dur', 'shim', 'grp', 'bri']}

def add(pos, nrm, col, cls, t0, dur, shim, grp, bri):
    n = len(pos)
    for k, v in zip(bag.keys(), [pos, nrm, col, cls, t0, dur, np.full(n, shim), np.full(n, grp), bri]):
        bag[k].append(np.asarray(v))

def cls_draw(n, p0, p1):
    u = rng.random(n)
    return np.where(u < p0, 0, np.where(u < p0 + p1, 1, 2))

# ---------------- 1. Rose Head (48 Layered Whorls) ----------------
def build_head(Ntot):
    P = 48
    ks = np.arange(P)
    tp = ks / (P - 1)
    ang0 = ks * 2.399963 + rng.normal(0, 0.06, P)

    # 3-tier geometry: inner tight bud cone -> mid cupped whorls -> grand outer guard petals
    rhob = np.where(tp < 0.28, 0.03 + 0.12*tp, np.where(tp < 0.65, 0.06 + 0.25*tp, 0.22 + 0.20*tp))
    Hk = np.where(tp < 0.28, 0.82 + 0.10*tp, np.where(tp < 0.65, 0.88 + 0.15*tp, 0.94 + 0.12*tp)) * rng.uniform(0.95, 1.05, P)
    zb = np.where(tp < 0.28, 0.20 - 0.15*tp, np.where(tp < 0.65, 0.16 - 0.25*tp, -0.05 - 0.18*(tp - 0.65)/0.35))
    alpha = np.where(tp < 0.28, 3.4 + 0.4*(1 - tp/0.28), np.where(tp < 0.65, 2.8 + 0.4*(1 - (tp - 0.28)/0.37), 2.2 + 0.4*(1 - (tp - 0.65)/0.35)))
    tilt0 = np.where(tp < 0.28, 0.02 + 0.18*tp, np.where(tp < 0.65, 0.18 + 0.35*(tp - 0.28)/0.37, 0.48 + 0.32*(tp - 0.65)/0.35)) + rng.normal(0, 0.03, P)
    bend = np.where(tp < 0.28, -0.15 + 0.40*tp, np.where(tp < 0.65, 0.10 + 0.75*(tp - 0.28)/0.37, 0.65 + 0.60*(tp - 0.65)/0.35)) + rng.normal(0, 0.04, P)

    area = (Hk * alpha * (rhob + 0.35)) ** 0.85
    nk = np.round(Ntot * area / area.sum()).astype(int)
    sg = np.linspace(0, 1, 257)
    ds = 1 / 256
    out = []

    for k in range(P):
        tilt = tilt0[k] + bend[k] * sg**2.2
        sT, cT = np.sin(tilt), np.cos(tilt)
        rho_t = np.maximum((rhob[k] + Hk[k]*np.concatenate([[0], np.cumsum(0.5*(sT[1:] + sT[:-1])*ds)])) * 0.84, 0.025)
        z_t = zb[k] + Hk[k]*np.concatenate([[0], np.cumsum(0.5*(cT[1:] + cT[:-1])*ds)])
        ph = rng.uniform(0, TAU)
        n = nk[k]
        s = rng.random(n)
        w = rng.uniform(-1, 1, n)
        m = rng.random(n)
        a = m < 0.07; w[a] = np.sign(w[a])
        b = (m >= 0.07) & (m < 0.16); s[b] = 1.0

        def F(s, w):
            wf = (0.50 + 0.50*np.sin(np.pi*(0.15 + 0.70*s))) * np.sqrt(np.clip(1 - 0.75*s**6, 0.05, 1))
            we = w * wf
            r = np.interp(s, sg, rho_t) + 0.045*s**2*we**2 + 0.022*s**2*np.sin(2.5*np.pi*we + ph)
            z = np.interp(s, sg, z_t) - 0.025*s*np.abs(we) + 0.025*s**3*np.sin(3.0*np.pi*we + ph*1.7)
            th = ang0[k] + we*alpha[k]/2
            return np.stack([r*np.cos(th), r*np.sin(th), z], -1)

        e = 0.012
        Pp = F(s, w)
        dPs = F(np.clip(s + e, 0, 1), w) - F(np.clip(s - e, 0, 1), w)
        dPw = F(s, np.clip(w + e, -1, 1)) - F(s, np.clip(w - e, -1, 1))
        N = nz(np.cross(dPs, dPw))
        Pp = Pp + N * rng.normal(0, 0.0045, (n, 1))

        shade = (0.15 + 0.85*s**1.1)*(0.50 + 0.50*tp[k]**0.5) + rng.normal(0, 0.06, n)
        col = mix(C_DARK, C_MID, np.clip(shade*1.1, 0, 1))
        col = mix(col, C_HI, np.clip((s - 0.78)*3.2, 0, 1)*0.5)
        if tp[k] > 0.65:
            col = mix(col, PINK_SOFT, np.clip((s - 0.85)*4.0, 0, 1)*0.4)
        col[:, 0] += rng.normal(0, 0.035)

        rm = m < 0.16
        col[rm] = mix(col[rm], C_HI, np.full(rm.sum(), 0.55))
        rim = rm & (rng.random(n) < 0.22)
        col = mix(col, GOLD, rim * 0.55)
        bri = np.full(n, rng.uniform(0.78, 1.18))
        bri[rm] *= 1.4
        sp = rng.random(n) < 0.03
        col[sp] = GOLD
        bri[sp] = 1.5

        # Inner petals form first, mid then outer bloom smoothly (all complete before t=10.5s)
        t0 = 3.0 + 3.8*tp[k] + 0.4*rng.random(n)
        dur = 2.4 + 0.5*rng.random(n)
        out.append((Pp, N, col, cls_draw(n, 0.9, 0.1), t0, dur, bri))

    cat = [np.concatenate([o[i] for o in out]) for i in range(7)]
    base = np.array([0, 0, -0.18])
    a, b = 0.10, -0.05
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R = Ry@Rx
    cat[0] = (cat[0] - base)@R.T + base
    cat[1] = cat[1]@R.T
    add(cat[0], cat[1], cat[2], cat[3], cat[4], cat[5], 1.0, 0, cat[6])

def build_receptacle(n):
    v = nz(rng.normal(size=(n*2, 3)))
    v = v[v[:, 2] < 0.25][:n]
    n = len(v)
    pos = np.array([0, 0, -0.16]) + v*np.array([0.24, 0.24, 0.14])
    col = mix(GREEN_D*1.3, GREEN_L, rng.random(n)*0.5)
    add(pos, nz(v), col, cls_draw(n, 0.85, 0.13), 3.0 + 0.5*rng.random(n), 2.2 + 0.5*rng.random(n), 0.4, 3, np.ones(n))

# ---------------- 2. Leaves & Calyx Sepals ----------------
def leaf(origin, ang, L, Wm, elev0, droop, n, t0b, t0s, grp, bri_s=1.0, tipcurve=1.6):
    dirh = np.array([np.cos(ang), np.sin(ang), 0.0])
    lat = np.array([-np.sin(ang), np.cos(ang), 0.0])
    ug = np.linspace(0, 1, 200)
    elev = elev0 - droop * ug**tipcurve
    T = np.cos(elev)[:, None]*dirh + np.sin(elev)[:, None]*np.array([0, 0, 1.0])
    mid = origin + L*(np.cumsum(T, axis=0) - T[0])/199
    typ = rng.random(n)
    u = rng.random(n); v = rng.uniform(-1, 1, n)
    mr = typ < 0.10; v[mr] = 0
    ed = (typ >= 0.10) & (typ < 0.24); v[ed] = np.sign(rng.random(ed.sum()) - 0.5)
    vn = (typ >= 0.24) & (typ < 0.40)
    q = rng.random(vn.sum()); k = rng.integers(1, 8, vn.sum())
    u[vn] = np.clip(k/9 + 0.28*q, 0, 1); v[vn] = np.sign(rng.random(vn.sum()) - 0.5)*q
    wp = Wm*np.sin(np.pi*np.clip(u, 0, 1)**0.7)**0.85 * (1 + 0.05*np.sin(TAU*u*13)*ed)
    mu = np.stack([np.interp(u, ug, mid[:, i]) for i in range(3)], -1)
    Tu = nz(np.stack([np.interp(u, ug, T[:, i]) for i in range(3)], -1))
    Nn = nz(np.cross(Tu, lat))
    pos = mu + lat*(v*wp)[:, None] + Nn*(0.45*wp*np.abs(v) + 0.012*np.sin(9*u + ang)*np.abs(v))[:, None]
    pos += rng.normal(0, 0.003, pos.shape)
    nrm = nz(Nn - 0.45*np.sign(v)[:, None]*lat)
    g = 0.25 + 0.75*u
    col = mix(GREEN_D, GREEN_L, np.clip(g*0.7 + rng.normal(0, 0.1, n), 0, 1))
    col = mix(col, GREEN_L*1.5, (mr | vn)*0.55)
    col = mix(col, GOLD*0.7, (ed & (rng.random(n) < 0.12))*0.5)
    t0 = t0b + t0s*u + 0.3*rng.random(n)
    add(pos, nrm, col*0.88, cls_draw(n, 0.85, 0.13), t0, 2.2 + 0.4*rng.random(n), 0.35, grp, bri_s*np.ones(n))

# ---------------- 3. Stem & Thorns ----------------
P0, P1, P2, P3 = [np.array(p) for p in [(0, 0, -0.18), (0.02, 0.07, -0.80), (-0.28, -0.09, -1.65), (-0.10, 0.05, -2.60)]]

def bez(u):
    u = np.asarray(u)[..., None]
    return (1 - u)**3*P0 + 3*(1 - u)**2*u*P1 + 3*(1 - u)*u**2*P2 + u**3*P3

def bezd(u):
    u = np.asarray(u)[..., None]
    return 3*(1 - u)**2*(P1 - P0) + 6*(1 - u)*u*(P2 - P1) + 3*u**2*(P3 - P2)

def build_stem(n):
    u = rng.random(n); phi = rng.uniform(0, TAU, n)
    c = bez(u); T = nz(bezd(u)); h = np.array([1.0, 0.3, 0.0])
    n1 = nz(np.cross(T, h)); n2 = nz(np.cross(T, n1))
    r = (0.046 - 0.012*u)[:, None]
    rad = np.cos(phi)[:, None]*n1 + np.sin(phi)[:, None]*n2
    pos = c + rad*r + rng.normal(0, 0.002, (n, 3))
    col = mix(GREEN_D*1.2, GREEN_L*0.8, np.clip(0.4 + 0.5*np.cos(phi - 1.0) + rng.normal(0, 0.1, n), 0, 1))
    add(pos, rad, col, cls_draw(n, 0.85, 0.13), 2.4 + 2.0*u + 0.3*rng.random(n), 2.2 + 0.4*rng.random(n), 0.4, 1, np.ones(n))
    for ut, ph in [(0.18, 0.5), (0.32, 3.8), (0.46, 2.0), (0.6, 5.2), (0.74, 1.0), (0.86, 4.0)]:
        m = 140; tt = rng.random(m); a = rng.uniform(0, TAU, m)
        cc = bez(ut); T0 = nz(bezd(ut)); n1 = nz(np.cross(T0, [1, 0.3, 0])); n2 = nz(np.cross(T0, n1))
        d = np.cos(ph)*n1 + np.sin(ph)*n2; d2 = nz(d + 0.5*np.array([0, 0, 1.0]))
        side = nz(np.cross(d2, T0))
        pos = cc + d*0.04 + d2*(0.09*tt)[:, None] + (np.cos(a)[:, None]*side + np.sin(a)[:, None]*T0)*(0.017*(1 - tt))[:, None]
        col = np.tile(np.array([0.45, 0.18, 0.05]), (m, 1))*(0.6 + 0.6*(1 - tt))[:, None]
        add(pos, np.tile(nz(d2 + side*0.3), (m, 1)), col, cls_draw(m, 0.9, 0.1), 2.4 + 2.0*ut + 0.4 + 0.3*rng.random(m), 2.0*np.ones(m), 0.3, 1, np.ones(m))

# ---------------- 4. Atmospheric Aura / Golden Dust ----------------
def build_aura(n):
    col = np.tile(GOLD, (n, 1)); u = rng.random(n)
    col[u > 0.68] = np.array([1.0, 0.36, 0.10])
    col[u > 0.90] = np.array([0.9, 0.05, 0.08])
    col *= rng.uniform(0.7, 1.0, (n, 1))
    add(np.zeros((n, 3)), np.zeros((n, 3)), col, cls_draw(n, 0.35, 0.42), np.zeros(n), np.ones(n), 0.0, 4, np.full(n, 0.25))

# Build complete 3D scene elements
t_build = time.time()
build_head(105000)
build_receptacle(2500)
build_stem(7500)

leaf_configs = [
    (0.4, 0.26, 0.58, 0.13, 0.75, 0.95, 4500, 3.8),
    (2.9, 0.40, 0.54, 0.12, 0.70, 0.90, 4200, 4.4),
    (1.4, 0.56, 0.48, 0.11, 0.65, 0.85, 3800, 5.0),
    (4.5, 0.72, 0.42, 0.10, 0.60, 0.80, 3500, 5.6),
]
for ang_, ut, L, Wm, el, dr, n_lf, t0_lf in leaf_configs:
    org = bez(ut) + np.array([np.cos(ang_), np.sin(ang_), 0])*0.05
    leaf(org, ang_, L, Wm, el, dr, n_lf, t0_lf, 1.0, 2)

for i in range(5):
    a = i*TAU/5 + 0.35 + rng.normal(0, 0.08)
    org = np.array([np.cos(a)*0.15, np.sin(a)*0.15, -0.15])
    leaf(org, a, 0.50, 0.068, 0.40, 1.55, 700, 3.2, 0.4, 3, 1.15, 1.25)

NAURA = 4000
build_aura(NAURA)

for k in bag: bag[k] = np.concatenate(bag[k])
POS, NRM, COL, CLS, T0, DURp, SHIM, GRP, BRI = [bag[k] for k in ['pos', 'nrm', 'col', 'cls', 't0', 'dur', 'shim', 'grp', 'bri']]
N = len(POS)
isA = GRP == 4; iF = np.where(~isA)[0]; iA = np.where(isA)[0]
NF = len(iF)

TA = np.where(isA, 0.3 + 2.0*rng.random(N)**0.45, np.maximum(1.0, T0 - 0.3 - 1.2*rng.random(N)))
T0 = np.where(isA, 0, np.maximum(T0, TA + 0.3))
TWF = rng.uniform(0.5, 1.4, N); TWP = rng.uniform(0, TAU, N)
SPP = rng.uniform(0, TAU, N)

Pf = POS[iF]
th_t = np.arctan2(Pf[:, 1], Pf[:, 0]); r_t = np.hypot(Pf[:, 0], Pf[:, 1]); z_t = Pf[:, 2]
v = nz(rng.normal(size=(NF, 3))); rho = 0.7 + 1.8*rng.random(NF)**0.8
S = np.array([0, 0, -0.5]) + v*rho[:, None]*np.array([1, 1, 0.9])
th_s = np.arctan2(S[:, 1], S[:, 0]); r_s = np.hypot(S[:, 0], S[:, 1]); z_s = S[:, 2]
dth = ((th_t - th_s) % TAU) + TAU*rng.integers(0, 3, NF)
om = rng.uniform(0.15, 0.45, NF)
dr_amp = rng.uniform(0.04, 0.12, NF)[:, None]; dr_f = rng.uniform(0.3, 0.9, (NF, 3)); dr_p = rng.uniform(0, TAU, (NF, 3))
t0F = T0[iF]; duF = DURp[iF]; shF = SHIM[iF]

r_f = np.clip(rng.normal(1.05, 0.33, NAURA), 0.65, 2.2); z_f = rng.normal(-0.05, 0.55, NAURA)
th_a = rng.uniform(0, TAU, NAURA); om_a = rng.uniform(0.2, 0.55, NAURA)/(0.6 + r_f)
r_a0 = rng.uniform(2.4, 4.2, NAURA); z_a0 = rng.uniform(-2.2, 1.6, NAURA); ph_a = rng.uniform(0, TAU, NAURA)

def positions(t):
    out = np.empty((N, 3))
    p = np.clip((t - t0F)/duF, 0, 1)
    e = sm(p)**0.55
    th = th_s + dth*e + om*(0.6*t + 0.04*t*t)*(1 - e)
    r = r_s + (r_t - r_s)*sm(p**0.9)**0.55
    z = z_s + (z_t - z_s)*e
    g = shF * sm((t - 6.0)/3.0)
    th = th + g*0.03*np.sin(1.1*t + 6*z_t + 1.5*th_t)
    r = r*(1 + g*0.010*np.sin(1.7*t + 4*z_t + 0.7*th_t))
    x = r*np.cos(th); y = r*np.sin(th)
    drift = dr_amp*np.sin(dr_f*t + dr_p)*(1 - e)[:, None]
    jit = (g*0.004)[:, None]*np.sin(np.stack([2.9, 3.4, 2.3])*t + dr_p*3)
    out[iF] = np.stack([x, y, z], -1) + drift + jit
    k = sm((t - 1.0)/7.0)
    ra = r_f + (r_a0 - r_f)*(1 - k)
    za = z_f + (z_a0 - z_f)*(1 - k) + 0.07*np.sin(0.9*t + ph_a)
    tha = th_a + om_a*t*(1.0 + 1.2*(1 - k))
    out[iA] = np.stack([ra*np.cos(tha), ra*np.sin(tha), za], -1)
    return out

# Camera Projection Settings
F_PIX = 2300.0
TARGET = np.array([0, 0, -0.78])
GW, GH = W//6, H//6
GAIN = np.array([1.0, 3.2, 9.0])
GB = 1.7
DITHER = (np.random.default_rng(3).random((H, W, 1)).astype(np.float32) - 0.5)/255.0

def render(t, exposure=1.0, yaw=None, pitch=None, Dd=None):
    # Smooth continuous 3D camera orbit that showcases the flower from multiple perspectives
    yaw = 0.20 + 2.8*sm((t - 6.0)/13.5) if yaw is None else yaw
    pitch = 0.68 - 0.08*sm(t/18.0) if pitch is None else pitch
    D = 8.0 - 1.5*sm((t - 2.0)/10.0) if Dd is None else Dd

    c, s = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    M = np.array([[1, 0, 0], [0, sp, cp], [0, cp, -sp]])@Rz
    dts = [0.0, 0.045, 0.09]
    scr = []
    for d in dts:
        P = positions(max(t - d, 0.0))
        q = (P - TARGET)@M.T
        dep = q[:, 2] + D
        scr.append((W/2 + F_PIX*q[:, 0]/dep, H/2 - F_PIX*q[:, 1]/dep, dep))
    x0, y0, dep = scr[0]

    cell = np.clip((y0//6).astype(np.int64), 0, GH - 1)*GW + np.clip((x0//6).astype(np.int64), 0, GW - 1)
    zmin = np.full(GW*GH, 1e9)
    np.minimum.at(zmin, cell, dep)
    occ = np.exp(-np.maximum(0, dep - zmin[cell] - 0.022)/0.03)
    occ = np.where(isA, np.maximum(occ, 0.35), np.maximum(occ, 0.06))

    n = NRM@M.T
    L = nz(np.array([-0.45, 0.55, -0.7]))
    V = np.array([0, 0, -1.0])
    Hh = nz(L + V)
    nl = np.abs(n@L); nv = np.abs(n@V); nh = np.abs(n@Hh)
    lit = 0.12 + 1.15*nl**1.3 + 0.5*(1 - nv)**2
    spec = nh**40
    lit = np.where(isA, 1.0, lit)

    alpha = sm((t - TA)/0.8)
    PP = np.ones(N)
    PP[iF] = np.clip((t - t0F)/duF, 0, 1)
    act = 0.55 + 0.45*sm(PP*3)
    tw = 0.78 + 0.22*np.sin(TAU*TWF*t + TWP)
    gsh = sm((t - 6.0)/2.5)*np.where(isA, 0.5, SHIM)
    zt = POS[:, 2]; tht = np.arctan2(POS[:, 1], POS[:, 0])
    spk = np.clip(np.sin(2.3*t - 4*zt - 2.2*tht + SPP), 0, 1)**8 * gsh
    col = COL + (GOLD - COL)*(0.5*spk)[:, None]
    col = col + np.array([1.0, 0.45, 0.35])*(spec*0.9*(~isA))[:, None]
    dz = np.abs(dep - D)
    cls = np.minimum(2, CLS + (dz > 1.1) + (dz > 2.2))
    att = np.exp(-0.12*np.maximum(dep - D, 0))
    E = col*(BRI*act*alpha*tw*(1 + 1.8*spk)*occ*lit*att*GB*GAIN[cls])[:, None]

    wts = [0.6, 0.2, 0.1, 0.06, 0.04]
    xs = [scr[0][0], 0.5*(scr[0][0] + scr[1][0]), scr[1][0], 0.5*(scr[1][0] + scr[2][0]), scr[2][0]]
    ys = [scr[0][1], 0.5*(scr[0][1] + scr[1][1]), scr[1][1], 0.5*(scr[1][1] + scr[2][1]), scr[2][1]]
    keep = np.where(alpha > 0.004)[0]
    il, wl = [], []
    for x, y, wj in zip(xs, ys, wts):
        x = x[keep]; y = y[keep]
        xi = np.floor(x).astype(np.int64); yi = np.floor(y).astype(np.int64)
        fx = x - xi; fy = y - yi
        inb = ((xi >= 1) & (xi < W - 2) & (yi >= 1) & (yi < H - 2)).astype(np.float32)
        xi = np.clip(xi, 1, W - 3); yi = np.clip(yi, 1, H - 3)
        base = cls[keep]*(W*H) + yi*W + xi
        for dx, dy, ww in ((0, 0, (1 - fx)*(1 - fy)), (1, 0, fx*(1 - fy)), (0, 1, (1 - fx)*fy), (1, 1, fx*fy)):
            il.append(base + dy*W + dx)
            wl.append(ww*wj*inb)
    idx = np.concatenate(il); wg = np.concatenate(wl); rep = len(il)
    chans = []
    for ch in range(3):
        b = np.bincount(idx, weights=wg*np.tile(E[keep, ch], rep), minlength=3*W*H).astype(np.float32)
        chans.append(b.reshape(3, H, W))
    base_img = None
    for ci, sig in enumerate([0.8, 1.6, 3.4]):
        im = np.dstack([chans[0][ci], chans[1][ci], chans[2][ci]])
        im = cv2.GaussianBlur(im, (0, 0), sig)
        base_img = im if base_img is None else base_img + im
    small = cv2.resize(base_img, (W//4, H//4), interpolation=cv2.INTER_AREA)
    bl = 0.30*cv2.GaussianBlur(small, (0, 0), 2.5) + 0.22*cv2.GaussianBlur(small, (0, 0), 7) + 0.16*cv2.GaussianBlur(small, (0, 0), 18)
    bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)
    img = (base_img + bl)*exposure
    out = (1 - np.exp(-img))**(1/1.7)
    out = np.clip(out + DITHER, 0, 1)
    return (out*255 + 0.5).astype(np.uint8)

def render_frame_idx(i):
    return render(i / FPS).tobytes()

if __name__ == '__main__':
    print('3D Particle Rose Model:', N, 'particles, built in', round(time.time() - t_build, 2), 's', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_times = sys.argv[2:] if len(sys.argv) > 2 else ['5.0', '10.0', '16.0']
        for tt in test_times:
            t1 = time.time(); fr = render(float(tt))
            out_png = f'test_{tt}.png'
            cv2.imwrite(out_png, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            print(f'{tt}s -> {out_png} in {round(time.time() - t1, 2)}s', flush=True)
    else:
        out = 'particle_rose.mp4'
        if len(sys.argv) >= 3:
            out = sys.argv[2]
        elif len(sys.argv) == 2:
            out = sys.argv[1]
        if os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass
        nf = int(DUR * FPS)
        print(f'Rendering full cinematic video: {DUR}s ({nf} frames at {FPS} FPS, {W}x{H}) to {out}...', flush=True)
        ff = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out
        ], stdin=subprocess.PIPE)
        t1 = time.time()
        num_workers = min(6, os.cpu_count() or 4)
        print(f'Using {num_workers} parallel workers for rendering...', flush=True)
        with mp.Pool(processes=num_workers) as pool:
            for i, frame_bytes in enumerate(pool.imap(render_frame_idx, range(nf), chunksize=2)):
                ff.stdin.write(frame_bytes)
                if (i + 1) % 20 == 0 or i == nf - 1:
                    elapsed = time.time() - t1
                    fps_speed = (i + 1) / elapsed
                    eta = (nf - (i + 1)) / fps_speed if fps_speed > 0 else 0
                    print(f'Frame {i+1}/{nf} ({round((i+1)/nf*100, 1)}%) - {fps_speed:.2f} fps - elapsed: {round(elapsed)}s - ETA: {round(eta)}s', flush=True)
        ff.stdin.close()
        ff.wait()
        print(f'Done! Rendered {nf} frames in {round(time.time() - t1)}s. Saved to {out}', flush=True)
