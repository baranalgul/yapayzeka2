import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# 1) Motor Parametreleri
# -----------------------------------------
R, L = 1.0, 0.5
Kt, Kb = 0.01, 0.01
J, B = 0.01, 0.001
Vmax = 24.0

# -----------------------------------------
# 2) Üyelik Fonksiyonu Yardımcıları
# -----------------------------------------
def tri(x, a, b, c):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[(a < x) & (x <= b)] = (x[(a < x) & (x <= b)] - a) / (b - a + 1e-12)
    y[(b < x) & (x < c)] = (c - x[(b < x) & (x < c)]) / (c - b + 1e-12)
    y[x == b] = 1
    return y

# Üyelik kümelerinin tek yerde tanımı
MF_e  = {'NB':(-150,-100,-50), 'NS':(-100,-50,0), 'Z':(-10,0,10), 'PS':(0,50,100), 'PB':(50,100,150)}
MF_de = {'N':(-50,-25,0), 'Z':(-10,0,10), 'P':(0,25,50)}
MF_u  = {'N':(-Vmax, -0.7*Vmax, 0), 'Z':(-4,0,4), 'P':(0, 0.7*Vmax, Vmax)}

# Kural tablosu (e x de)
RULES = [
    ['N','N','N'],  # NB
    ['N','N','Z'],  # NS
    ['N','Z','P'],  # Z
    ['Z','P','P'],  # PS
    ['P','P','P']   # PB
]

e_labels  = list(MF_e.keys())
de_labels = list(MF_de.keys())

# -----------------------------------------
# 3) Fuzzy Controller (tek fonksiyonda toplandı)
# -----------------------------------------
def fuzzy_control(e, de, u_disc=np.linspace(-Vmax, Vmax, 801)):
    # 1 - fuzzify
    mu_e  = {lab: tri([e],  *MF_e[lab])[0] for lab in MF_e}
    mu_de = {lab: tri([de], *MF_de[lab])[0] for lab in MF_de}

    # 2 - inference + aggregation
    aggr = np.zeros_like(u_disc)
    for i, e_lab in enumerate(e_labels):
        for j, de_lab in enumerate(de_labels):
            fire = min(mu_e[e_lab], mu_de[de_lab])
            if fire == 0: 
                continue
            mf = tri(u_disc, *MF_u[RULES[i][j]])
            aggr = np.maximum(aggr, np.minimum(mf, fire))

    # 3 - defuzz (centroid)
    if np.sum(aggr) == 0:
        return 0.0
    u = np.sum(u_disc * aggr) / np.sum(aggr)
    return float(np.clip(u, -Vmax, Vmax))

# -----------------------------------------
# 4) Motor Modeli + RK4
# -----------------------------------------
def motor_dx(x, u, TL=0):
    i, w = x
    di = (-R*i - Kb*w + u) / L
    dw = (-B*w + Kt*i - TL) / J
    return np.array([di, dw])

def rk4(x, u, dt, TL=0):
    k1 = motor_dx(x, u, TL)
    k2 = motor_dx(x + 0.5*dt*k1, u, TL)
    k3 = motor_dx(x + 0.5*dt*k2, u, TL)
    k4 = motor_dx(x + dt*k3, u, TL)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# -----------------------------------------
# 5) Simülasyon
# -----------------------------------------
def simulate(ref, T=5, dt=0.001, TL=None):
    t = np.arange(0, T+dt, dt)
    x = np.array([0.0, 0.0])
    
    w_hist, u_hist, e_hist, ref_hist = [], [], [], []
    prev_e = 0

    for tk in t:
        r = ref(tk)
        e = r - x[1]
        de = e - prev_e
        prev_e = e

        u = fuzzy_control(e, de)
        TLk = TL(tk) if TL else 0.0
        x = rk4(x, u, dt, TLk)

        ref_hist.append(r)
        w_hist.append(x[1])
        u_hist.append(u)
        e_hist.append(e)

    return t, np.array(w_hist), np.array(u_hist), np.array(e_hist), np.array(ref_hist)

# -----------------------------------------
# 6) Örnek Çalıştırma
# -----------------------------------------
if __name__ == "__main__":
    ref = lambda t: 100*(t/0.2) if t<0.2 else 100
    TL  = lambda t: 0.002 if t>=2 else 0

    t, w, u, e, ref_sig = simulate(ref, T=5, dt=0.001, TL=TL)

    # Grafikler
    plt.figure(figsize=(10,8))

    plt.subplot(3,1,1)
    plt.plot(t, ref_sig, '--', label='Referans')
    plt.plot(t, w, label='Hız')
    plt.ylabel("w (rad/s)")
    plt.grid(); plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, u); plt.ylabel("u (V)")
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(t, e); plt.ylabel("Hata")
    plt.xlabel("Zaman (s)")
    plt.grid()

    plt.tight_layout()
    plt.show()
