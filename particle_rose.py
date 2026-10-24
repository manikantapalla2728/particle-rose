import numpy as np, cv2, sys, subprocess, time
cv2.setNumThreads(2)
W, H, FPS, DUR = 1080, 1920, 30, 15.0
TAU = 2*np.pi
rng = np.random.default_rng(11)

def sm(x):
    x = np.clip(x, 0, 1); return x*x*x*(x*(x*6-15)+10)
def nz(v): return v/(np.linalg.norm(v, axis=-1, keepdims=True)+1e-9)
def mix(a, b, m): return a*(1-m[:, None]) + b*m[:, None]

C_DARK=np.array([0.40,0.0,0.025]); C_MID=np.array([0.95,0.035,0.065]); C_HI=np.array([0.95,0.14,0.13])
GOLD=np.array([1.0,0.70,0.25]); GREEN_D=np.array([0.015,0.26,0.07]); GREEN_L=np.array([0.10,0.62,0.20])

bag = {k: [] for k in ['pos','nrm','col','cls','t0','dur','shim','grp','bri']}
def add(pos, nrm, col, cls, t0, dur, shim, grp, bri):
    n = len(pos)
    for k, v in zip(bag.keys(), [pos, nrm, col, cls, t0, dur, np.full(n, shim), np.full(n, grp), bri]):
        bag[k].append(np.asarray(v))

def cls_draw(n, p0, p1):
    u = rng.random(n); return np.where(u < p0, 0, np.where(u < p0+p1, 1, 2))

# ---------------- rose head ----------------
def build_head(Ntot):
    P = 38
    ks = np.arange(P); tp = ks/(P-1)
    ang0 = ks*2.399963 + rng.normal(0, 0.08, P)
    alpha = rng.uniform(2.3, 3.2, P)*(0.92+0.1*tp)
    Hk = (0.55+0.30*tp**0.8)*rng.uniform(0.93, 1.07, P)
    tilt0 = 0.05+0.55*tp**1.3 + rng.normal(0, 0.04, P)
    bend = -0.95+1.9*tp**0.9 + rng.normal(0, 0.05, P)
    rhob = 0.05+0.24*tp**0.9
    zb = 0.08-0.20*tp
    area = Hk*alpha*(rhob+0.22)
    nk = np.round(Ntot*area/area.sum()).astype(int)
    sg = np.linspace(0, 1, 257); ds = 1/256
    out = []
    for k in range(P):
        tilt = tilt0[k]+bend[k]*sg**2.2
        sT, cT = np.sin(tilt), np.cos(tilt)
        rho_t = np.maximum(rhob[k]+Hk[k]*np.concatenate([[0], np.cumsum(0.5*(sT[1:]+sT[:-1])*ds)]), 0.025)
        z_t = zb[k]+Hk[k]*np.concatenate([[0], np.cumsum(0.5*(cT[1:]+cT[:-1])*ds)])
        ph = rng.uniform(0, TAU); n = nk[k]
        s = rng.random(n); w = rng.uniform(-1, 1, n); m = rng.random(n)
        a = m < 0.07; w[a] = np.sign(w[a]); b = (m >= 0.07) & (m < 0.16); s[b] = 1.0
        def F(s, w):
            wf = (0.55+0.45*np.sin(np.pi*(0.15+0.75*s)))*np.sqrt(np.clip(1-0.8*s**8, 0.05, 1))
            we = w*wf
            r = np.interp(s, sg, rho_t)+0.05*s**3*we**2+0.02*s**2*np.sin(2.4*np.pi*we+ph)
            z = np.interp(s, sg, z_t)+0.035*s**3*np.sin(3*np.pi*we+ph*2.1)
            th = ang0[k]+we*alpha[k]/2
            return np.stack([r*np.cos(th), r*np.sin(th), z], -1)
        e = 0.012
        Pp = F(s, w)
        dPs = F(np.clip(s+e, 0, 1), w)-F(np.clip(s-e, 0, 1), w)
        dPw = F(s, np.clip(w+e, -1, 1))-F(s, np.clip(w-e, -1, 1))
        N = nz(np.cross(dPs, dPw))
        Pp = Pp+N*rng.normal(0, 0.0045, (n, 1))
        shade = (0.12+0.88*s**1.1)*(0.55+0.45*tp[k]**0.5)+rng.normal(0, 0.07, n)
        col = mix(C_DARK, C_MID, np.clip(shade*1.1, 0, 1))
        col = mix(col, C_HI, np.clip((s-0.8)*3, 0, 1)*0.5)
        col[:, 0] += rng.normal(0, 0.04)
        rm = m < 0.16
        col[rm] = mix(col[rm], C_HI, np.full(rm.sum(), 0.55))
        rim = rm & (rng.random(n) < 0.22)
        col = mix(col, GOLD, rim*0.55)
        bri = np.full(n, rng.uniform(0.75, 1.15)); bri[rm] *= 1.4
        sp = rng.random(n) < 0.03
        col[sp] = GOLD; bri[sp] = 1.5
        t0 = 2.2+3.0*tp[k]+0.5*rng.random(n); dur = 2.8+0.8*rng.random(n)
        out.append((Pp, N, col, cls_draw(n, 0.9, 0.1), t0, dur, bri))
    cat = [np.concatenate([o[i] for o in out]) for i in range(7)]
    # tilt the head about its base
    base = np.array([0, 0, -0.2])
    a, b = 0.12, -0.07
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    R = Ry@Rx
    cat[0] = (cat[0]-base)@R.T+base; cat[1] = cat[1]@R.T
    add(cat[0], cat[1], cat[2], cat[3], cat[4], cat[5], 1.0, 0, cat[6])

def build_receptacle(n):
    v = nz(rng.normal(size=(n*2, 3))); v = v[v[:, 2] < 0.25][:n]; n = len(v)
    pos = np.array([0, 0, -0.17])+v*np.array([0.25, 0.25, 0.15])
    col = mix(GREEN_D*1.3, GREEN_L, rng.random(n)*0.5)
    add(pos, nz(v), col, cls_draw(n, 0.85, 0.13), 5.0+0.6*rng.random(n), 2.6+0.6*rng.random(n), 0.4, 3, np.ones(n))

# ---------------- leaves / sepals ----------------
def leaf(origin, ang, L, Wm, elev0, droop, n, t0b, t0s, grp, bri_s=1.0, tipcurve=1.6):
    dirh = np.array([np.cos(ang), np.sin(ang), 0.0]); lat = np.array([-np.sin(ang), np.cos(ang), 0.0])
    ug = np.linspace(0, 1, 200)
    elev = elev0-droop*ug**tipcurve
    T = np.cos(elev)[:, None]*dirh+np.sin(elev)[:, None]*np.array([0, 0, 1.0])
    mid = origin+L*(np.cumsum(T, axis=0)-T[0])/199
    typ = rng.random(n)
    u = rng.random(n); v = rng.uniform(-1, 1, n)
    mr = typ < 0.10; v[mr] = 0
    ed = (typ >= 0.10) & (typ < 0.24); v[ed] = np.sign(rng.random(ed.sum())-0.5)
    vn = (typ >= 0.24) & (typ < 0.40)
    q = rng.random(vn.sum()); k = rng.integers(1, 8, vn.sum())
    u[vn] = np.clip(k/9+0.28*q, 0, 1); v[vn] = np.sign(rng.random(vn.sum())-0.5)*q
    wp = Wm*np.sin(np.pi*np.clip(u, 0, 1)**0.7)**0.85*(1+0.05*np.sin(TAU*u*13)*ed)
    mu = np.stack([np.interp(u, ug, mid[:, i]) for i in range(3)], -1)
    Tu = nz(np.stack([np.interp(u, ug, T[:, i]) for i in range(3)], -1))
    Nn = nz(np.cross(Tu, lat))
    pos = mu+lat*(v*wp)[:, None]+Nn*(0.45*wp*np.abs(v)+0.012*np.sin(9*u+ang)*np.abs(v))[:, None]
    pos += rng.normal(0, 0.003, pos.shape)
    nrm = nz(Nn-0.45*np.sign(v)[:, None]*lat)
    g = 0.25+0.75*u
    col = mix(GREEN_D, GREEN_L, np.clip(g*0.7+rng.normal(0, 0.1, n), 0, 1))
    col = mix(col, GREEN_L*1.5, (mr | vn)*0.55)
    col = mix(col, GOLD*0.7, (ed & (rng.random(n) < 0.12))*0.5)
    t0 = t0b+t0s*u+0.4*rng.random(n)
    add(pos, nrm, col*0.85, cls_draw(n, 0.85, 0.13), t0, 2.4+0.5*rng.random(n), 0.35, grp, bri_s*np.ones(n))

# ---------------- stem ----------------
P0, P1, P2, P3 = [np.array(p) for p in [(0, 0, -0.20), (0.0, 0.06, -0.8), (-0.30, -0.1, -1.6), (-0.12, 0.05, -2.55)]]
def bez(u):
    u = np.asarray(u)[..., None]
    return (1-u)**3*P0+3*(1-u)**2*u*P1+3*(1-u)*u**2*P2+u**3*P3
def bezd(u):
    u = np.asarray(u)[..., None]
    return 3*(1-u)**2*(P1-P0)+6*(1-u)*u*(P2-P1)+3*u**2*(P3-P2)

def build_stem(n):
    u = rng.random(n); phi = rng.uniform(0, TAU, n)
    c = bez(u); T = nz(bezd(u)); h = np.array([1.0, 0.3, 0.0])
    n1 = nz(np.cross(T, h)); n2 = nz(np.cross(T, n1))
    r = (0.046-0.012*u)[:, None]
    rad = np.cos(phi)[:, None]*n1+np.sin(phi)[:, None]*n2
    pos = c+rad*r+rng.normal(0, 0.002, (n, 3))
    col = mix(GREEN_D*1.2, GREEN_L*0.8, np.clip(0.4+0.5*np.cos(phi-1.0)+rng.normal(0, 0.1, n), 0, 1))
    add(pos, rad, col, cls_draw(n, 0.85, 0.13), 4.2+2.2*u+0.4*rng.random(n), 2.4+0.4*rng.random(n), 0.4, 1, np.ones(n))
    # thorns
    for ut, ph in [(0.18, 0.5), (0.32, 3.8), (0.46, 2.0), (0.6, 5.2), (0.74, 1.0), (0.86, 4.0)]:
        m = 140; tt = rng.random(m); a = rng.uniform(0, TAU, m)
        cc = bez(ut); T0 = nz(bezd(ut)); n1 = nz(np.cross(T0, [1, 0.3, 0])); n2 = nz(np.cross(T0, n1))
        d = np.cos(ph)*n1+np.sin(ph)*n2; d2 = nz(d+0.5*np.array([0, 0, 1.0]))
        side = nz(np.cross(d2, T0))
        pos = cc+d*0.04+d2*(0.09*tt)[:, None]+(np.cos(a)[:, None]*side+np.sin(a)[:, None]*T0)*(0.017*(1-tt))[:, None]
        col = np.tile(np.array([0.45, 0.18, 0.05]), (m, 1))*(0.6+0.6*(1-tt))[:, None]
        add(pos, np.tile(nz(d2+side*0.3), (m, 1)), col, cls_draw(m, 0.9, 0.1), 4.2+2.2*ut+0.5+0.3*rng.random(m), 2.0*np.ones(m), 0.3, 1, np.ones(m))

# ---------------- aura ----------------
AURA = []
def build_aura(n):
    col = np.tile(GOLD, (n, 1)); u = rng.random(n)
    col[u > 0.68] = np.array([1.0, 0.36, 0.10]); col[u > 0.90] = np.array([0.9, 0.05, 0.08])
    col *= rng.uniform(0.7, 1.0, (n, 1))
    add(np.zeros((n, 3)), np.zeros((n, 3)), col, cls_draw(n, 0.35, 0.42), np.zeros(n), np.ones(n), 0.0, 4, np.full(n, 0.24))

t_build = time.time()
build_head(100000); build_receptacle(2500)
build_stem(7000)
for ang_, ut, L, Wm, el, dr, side in [(0.3, 0.30, 0.55, 0.12, 0.75, 0.95, 1), (2.9, 0.52, 0.46, 0.11, 0.7, 0.9, 1)]:
    org = bez(ut)+np.array([np.cos(ang_), np.sin(ang_), 0])*0.05
    leaf(org, ang_, L, Wm, el, dr, 6500, 5.6, 1.5, 2)
for i in range(5):
    a = i*TAU/5+0.4+rng.normal(0, 0.1)
    org = np.array([np.cos(a)*0.17, np.sin(a)*0.17, -0.13])
    leaf(org, a, 0.46, 0.065, 0.35, 1.45, 600, 5.2, 0.4, 3, 1.1, 1.2)
NAURA = 3500; build_aura(NAURA)
for k in bag: bag[k] = np.concatenate(bag[k])
POS, NRM, COL, CLS, T0, DURp, SHIM, GRP, BRI = [bag[k] for k in ['pos','nrm','col','cls','t0','dur','shim','grp','bri']]
N = len(POS)
isA = GRP == 4; iF = np.where(~isA)[0]; iA = np.where(isA)[0]
NF = len(iF)
# appearance
TA = np.where(isA, 0.3+2.0*rng.random(N)**0.45, np.maximum(1.2, T0-0.3-1.4*rng.random(N)))
T0 = np.where(isA, 0, np.maximum(T0, TA+0.4))
TWF = rng.uniform(0.5, 1.4, N); TWP = rng.uniform(0, TAU, N)
SPP = rng.uniform(0, TAU, N)
# formers
Pf = POS[iF]
th_t = np.arctan2(Pf[:, 1], Pf[:, 0]); r_t = np.hypot(Pf[:, 0], Pf[:, 1]); z_t = Pf[:, 2]
v = nz(rng.normal(size=(NF, 3))); rho = 0.7+1.8*rng.random(NF)**0.8
S = np.array([0, 0, -0.5])+v*rho[:, None]*np.array([1, 1, 0.9])
th_s = np.arctan2(S[:, 1], S[:, 0]); r_s = np.hypot(S[:, 0], S[:, 1]); z_s = S[:, 2]
dth = ((th_t-th_s) % TAU)+TAU*rng.integers(0, 3, NF)
om = rng.uniform(0.15, 0.45, NF)
dr_amp = rng.uniform(0.04, 0.12, NF)[:, None]; dr_f = rng.uniform(0.3, 0.9, (NF, 3)); dr_p = rng.uniform(0, TAU, (NF, 3))
t0F = T0[iF]; duF = DURp[iF]; shF = SHIM[iF]
# aura
r_f = np.clip(rng.normal(1.05, 0.33, NAURA), 0.65, 2.2); z_f = rng.normal(-0.05, 0.55, NAURA)
th_a = rng.uniform(0, TAU, NAURA); om_a = rng.uniform(0.2, 0.55, NAURA)/(0.6+r_f)
r_a0 = rng.uniform(2.4, 4.2, NAURA); z_a0 = rng.uniform(-2.2, 1.6, NAURA); ph_a = rng.uniform(0, TAU, NAURA)

def positions(t):
    out = np.empty((N, 3))
    p = np.clip((t-t0F)/duF, 0, 1); e = sm(p)**0.55
    th = th_s+dth*e+om*(0.6*t+0.04*t*t)*(1-e)
    r = r_s+(r_t-r_s)*sm(p**0.9)**0.55; z = z_s+(z_t-z_s)*e
    g = shF*sm((t-7.0)/3.0)
    th = th+g*0.03*np.sin(1.1*t+6*z_t+1.5*th_t)
    r = r*(1+g*0.010*np.sin(1.7*t+4*z_t+0.7*th_t))
    x = r*np.cos(th); y = r*np.sin(th)
    drift = dr_amp*np.sin(dr_f*t+dr_p)*(1-e)[:, None]
    jit = (g*0.004)[:, None]*np.sin(np.stack([2.9, 3.4, 2.3])*t+dr_p*3)
    out[iF] = np.stack([x, y, z], -1)+drift+jit
    k = sm((t-1.0)/7.0)
    ra = r_f+(r_a0-r_f)*(1-k); za = z_f+(z_a0-z_f)*(1-k)+0.07*np.sin(0.9*t+ph_a)
    tha = th_a+om_a*t*(1.0+1.2*(1-k))
    out[iA] = np.stack([ra*np.cos(tha), ra*np.sin(tha), za], -1)
    return out

# camera
F_PIX = 1450.0 if (W / H) > 1.0 else 2504.0
TARGET = np.array([0, 0, -0.48]) if (W / H) > 1.0 else np.array([0, 0, -0.85])
GW, GH = W//6, H//6
GAIN = np.array([1.0, 3.2, 9.0])
GB = 1.7
DITHER = (np.random.default_rng(3).random((H, W, 1)).astype(np.float32)-0.5)/255.0

def render(t, exposure=1.0, yaw=None, pitch=None, Dd=None):
    yaw = 0.25+3.0*sm(t/13.0) if yaw is None else yaw
    pitch = 0.52-0.10*sm(t/15.0) if pitch is None else pitch
    D = 8.0-1.7*sm(t/13.0) if Dd is None else Dd
    c, s = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); M = np.array([[1, 0, 0], [0, sp, cp], [0, cp, -sp]])@Rz
    dts = [0.0, 0.045, 0.09]
    scr = []
    for d in dts:
        P = positions(max(t-d, 0.0)); q = (P-TARGET)@M.T; dep = q[:, 2]+D
        scr.append((W/2+F_PIX*q[:, 0]/dep, H/2-F_PIX*q[:, 1]/dep, dep))
    x0, y0, dep = scr[0]
    # occlusion
    cell = np.clip((y0//6).astype(np.int64), 0, GH-1)*GW+np.clip((x0//6).astype(np.int64), 0, GW-1)
    zmin = np.full(GW*GH, 1e9); np.minimum.at(zmin, cell, dep)
    occ = np.exp(-np.maximum(0, dep-zmin[cell]-0.022)/0.03)
    occ = np.where(isA, np.maximum(occ, 0.35), np.maximum(occ, 0.06))
    # lighting
    n = NRM@M.T
    L = nz(np.array([-0.45, 0.55, -0.7])); V = np.array([0, 0, -1.0]); Hh = nz(L+V)
    nl = np.abs(n@L); nv = np.abs(n@V); nh = np.abs(n@Hh)
    lit = 0.12+1.15*nl**1.3+0.5*(1-nv)**2
    spec = nh**40
    lit = np.where(isA, 1.0, lit)
    # twinkle / sparkle
    alpha = sm((t-TA)/0.8)
    PP = np.ones(N); PP[iF] = np.clip((t-t0F)/duF, 0, 1)
    act = 0.55+0.45*sm(PP*3)
    tw = 0.78+0.22*np.sin(TAU*TWF*t+TWP)
    gsh = sm((t-7.0)/2.5)*np.where(isA, 0.5, SHIM)
    zt = POS[:, 2]; tht = np.arctan2(POS[:, 1], POS[:, 0])
    spk = np.clip(np.sin(2.3*t-4*zt-2.2*tht+SPP), 0, 1)**8*gsh
    col = COL+(GOLD-COL)*(0.5*spk)[:, None]
    col = col+np.array([1.0, 0.45, 0.35])*(spec*0.9*(~isA))[:, None]
    dz = np.abs(dep-D)
    cls = np.minimum(2, CLS+(dz > 1.1)+(dz > 2.2))
    att = np.exp(-0.12*np.maximum(dep-D, 0))
    E = col*(BRI*act*alpha*tw*(1+1.8*spk)*occ*lit*att*GB*GAIN[cls])[:, None]
    # splat
    wts = [0.6, 0.2, 0.1, 0.06, 0.04]
    xs = [scr[0][0], 0.5*(scr[0][0]+scr[1][0]), scr[1][0], 0.5*(scr[1][0]+scr[2][0]), scr[2][0]]
    ys = [scr[0][1], 0.5*(scr[0][1]+scr[1][1]), scr[1][1], 0.5*(scr[1][1]+scr[2][1]), scr[2][1]]
    keep = np.where(alpha > 0.004)[0]
    il, wl = [], []
    for x, y, wj in zip(xs, ys, wts):
        x = x[keep]; y = y[keep]
        xi = np.floor(x).astype(np.int64); yi = np.floor(y).astype(np.int64); fx = x-xi; fy = y-yi
        inb = ((xi >= 1) & (xi < W-2) & (yi >= 1) & (yi < H-2)).astype(np.float32)
        xi = np.clip(xi, 1, W-3); yi = np.clip(yi, 1, H-3)
        base = cls[keep]*(W*H)+yi*W+xi
        for dx, dy, ww in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            il.append(base+dy*W+dx); wl.append(ww*wj*inb)
    idx = np.concatenate(il); wg = np.concatenate(wl); rep = len(il)
    chans = []
    for ch in range(3):
        b = np.bincount(idx, weights=wg*np.tile(E[keep, ch], rep), minlength=3*W*H).astype(np.float32)
        chans.append(b.reshape(3, H, W))
    base_img = None
    for ci, sig in enumerate([0.8, 1.6, 3.4]):
        im = np.dstack([chans[0][ci], chans[1][ci], chans[2][ci]])
        im = cv2.GaussianBlur(im, (0, 0), sig)
        base_img = im if base_img is None else base_img+im
    small = cv2.resize(base_img, (W//4, H//4), interpolation=cv2.INTER_AREA)
    bl = 0.30*cv2.GaussianBlur(small, (0, 0), 2.5)+0.22*cv2.GaussianBlur(small, (0, 0), 7)+0.16*cv2.GaussianBlur(small, (0, 0), 18)
    bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)
    img = (base_img+bl)*exposure
    out = (1-np.exp(-img))**(1/1.7)
    out = np.clip(out+DITHER, 0, 1)
    return (out*255+0.5).astype(np.uint8)  # RGB order

def render_frame_idx(i):
    return render(i / FPS).tobytes()

if __name__ == '__main__':
    import os, multiprocessing as mp
    print('particles', N, 'build', round(time.time()-t_build, 1), 's', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_times = sys.argv[2:] if len(sys.argv) > 2 else ['5.0', '10.0']
        for tt in test_times:
            t1 = time.time(); fr = render(float(tt))
            out_png = f'test_{tt}.png'
            cv2.imwrite(out_png, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            print(f'{tt}s -> {out_png} in {round(time.time()-t1, 2)}s', flush=True)
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
        print(f'Rendering {DUR}s animation ({int(DUR*FPS)} frames at {FPS} FPS) to {out}...', flush=True)
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        nf = int(DUR*FPS); t1 = time.time()
        num_workers = min(6, os.cpu_count() or 4)
        print(f'Using {num_workers} parallel workers for rendering...', flush=True)
        with mp.Pool(processes=num_workers) as pool:
            for i, frame_bytes in enumerate(pool.imap(render_frame_idx, range(nf), chunksize=2)):
                ff.stdin.write(frame_bytes)
                if (i + 1) % 15 == 0 or i == nf - 1:
                    elapsed = time.time() - t1
                    fps_speed = (i + 1) / elapsed
                    eta = (nf - (i + 1)) / fps_speed if fps_speed > 0 else 0
                    print(f'Frame {i+1}/{nf} ({round((i+1)/nf*100, 1)}%) - {fps_speed:.2f} fps - elapsed: {round(elapsed)}s - ETA: {round(eta)}s', flush=True)
        ff.stdin.close(); ff.wait()
        print(f'Done! Rendered {nf} frames in {round(time.time()-t1)}s. Saved to {out}', flush=True)

