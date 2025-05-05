import numpy as np
from scipy.optimize import differential_evolution, fsolve, minimize
import math
import multiprocessing

# Konstanten
P_I = 25  # kW
n_I = 2450  # min^-1
i_ges_soll = 22.5
i_12_bounds = (5.25,6)
t_iges = 0.015
h = 180  # mm
K_A = 1.3
K_max = 2.3
alpha_0 = 0.7
t_b = 10

# Materialkonstanten 16MnCr5
R_pn = 695  # MPa
sigma_bWN = 500 # Mpa
sigma_bF_zul = 0.5 * R_pn #Mpa
sigma_bW_zul = 0.25 * sigma_bWN #Mpa

# Auswahlwerte
allowed_beta_34 = [12.5, 15]
allowed_mn = [2.25, 2.5, 2.75,3.0]
    

# Optimierungsgrenzen
bounds = [
    (17, 200), #z1
    (5.25,6), #i_12
    (i_ges_soll * (1 - t_iges), i_ges_soll * (1 + t_iges)), #i_ges
    (17, 200),  #z3
    (7.5, 17.5), #beta_12
    (0, len(allowed_beta_34)-1),  # beta_34_index
    (0, len(allowed_mn)-1),  # m_n_12
    (0, len(allowed_mn)-1),  # m_n_34
    (0.35, 0.75), (0.35, 0.75),  # psi_d
    (310, 525),  # sigma_Flim
]

intefrality= [
    True, 
    False, 
    False, 
    True, 
    False, 
    True, 
    True,
    True,
    False,
    False,
    False,
]

# Hilfsfunktion für Drehmoment & Wellendurchmesser
def berechne_welle(M_b, M_t):
    """ 
    Berechnet den Wellendurchmesser
    Args:
        M_b: Biegemoment in Nm
        M_t: Torsionsmoment in Nm
        sigma: zulässige Spannung in MPa
    Returns:
        Wellendurchmesser in mm
    """
    M_veq = K_A * np.sqrt(M_b** 2 + 0.75 * (alpha_0 * M_t) ** 2)
    M_vmax = K_max * np.sqrt(M_b**2 + 0.75 * M_t**2)
    d_stat = 2.17 * np.cbrt(M_vmax * 1000 / sigma_bF_zul)
    d_dyn = 2.17 * np.cbrt(M_veq * 1000 / sigma_bW_zul)
    return max(d_stat, d_dyn)

def calc_m_n_min(M_t, z, psi_d, beta, sigma_Flim):
    """
    Berechnet das minimale Modul m''
    Args:   
        M_t: Torsionsmoment der Welle in Nm 
        z: Zähnezahl der Welle
        psi_d: Übersetzungsverhältnis
        beta: Eingriffswinkel in Grad
    Returns:
        minimales Modul m''
    """
    beta_rad = math.radians(beta)
    return 1.85 * np.cbrt((M_t * 1000 * np.cos(beta_rad)**2) / (z**2 * psi_d * sigma_Flim))

def calc_d_z(z, m, beta):
    """
    Berechnet den Teilkreisdurchmesser
    Args:
        z: Zähnezahl
        m: Modul
        beta: Eingriffswinkel in Grad
    Returns:
        Teilkreisdurchmesser in mm
    """
    beta_rad = math.radians(beta)
    return (z * m) / np.cos(beta_rad)



# Zielfunktion
def objective(vars):
    z1 = int(vars[0])
    i_12 = vars[1]
    i_ges = vars[2]
    z3 = int(vars[3])
    beta_12 = vars[4]
    beta_34 = allowed_beta_34[int(vars[5])]
    m_n_12 = allowed_mn[int(vars[6])]    
    m_n_34 = allowed_mn[int(vars[7])]
    psi_d_12, psi_d_34 = vars[8:10]
    sigma_Flim = vars[10]

    error_sum = 0
    print(f"m_n_12: {m_n_12}, m_n_34: {m_n_34}")
    
    
    z2 = math.ceil(z1 * i_12)

    i_34 = i_ges / i_12

    z4 = math.ceil(z3 * i_34)


    error_sum += abs(np.gcd(z1, z2) + np.gcd(z3, z4))/100


    M_t_I = P_I * 1000 / (2 * np.pi * (n_I / 60)) # M_t_I
    M_t_II = M_t_I * i_12 # M_t_II
    M_t_III = M_t_I * i_ges # M_t_III

    M_t_1_eq = M_t_I * K_A
    M_t_3_eq = M_t_II * K_A 


    M_b_I = M_t_I
    M_b_II = 2 * M_t_II
    M_b_III = M_t_III

    d_I = berechne_welle(M_b_I, M_t_I)
    d_II = berechne_welle(M_b_II, M_t_II)
    d_III = berechne_welle(M_b_III, M_t_III)
    

    # Berechne überschlägiges Modul m''
    m_n_min_12 = calc_m_n_min(M_t_1_eq, z1, psi_d_12, beta_12, sigma_Flim)
    if m_n_12 < m_n_min_12:
        error_sum +=  (10000000* abs(m_n_12 - m_n_min_12)) **3
        


    d_1 = calc_d_z(z1, m_n_12, beta_12)
    d_2 = calc_d_z(z2, m_n_12, beta_12)

    m_n_min_34 = calc_m_n_min(M_t_3_eq, z3, psi_d_34, beta_34, sigma_Flim)

    
    if m_n_34 < m_n_min_34:
        error_sum +=  (100000000* abs(m_n_34 - m_n_min_34)) **3
    
    d_3 = calc_d_z(z3, m_n_34, beta_34)
    d_4 = calc_d_z(z4, m_n_34, beta_34)

 
    
   
    

    a_12 = (d_1 + d_2) / 2
    a_34 = (d_3 + d_4) / 2

    error_sum +=  (10000000* abs(a_12 - a_34)) **4

    d_k_1 = d_1 + 2 * m_n_12
    d_k_4 = d_4 + 2 * m_n_34

    s_a_1 = 2 +3*m_n_12 
    s_a_4 = 2 +3*m_n_34

    if d_k_1/2 * 2.5*s_a_1 + t_b > h:
        error_sum += (10000000 * abs(h - (d_k_1/2 * 2.5*s_a_1 + t_b))) ** 2
    if d_k_4/2 * 2.5*s_a_4 + t_b > h:
        error_sum += (10000000 * abs(h - (d_k_4/2 * 2.5*s_a_4 + t_b))) ** 2


    for df, di in [(d_1 - 2.5 * m_n_12, d_I), (d_2 - 2.5 * m_n_12, d_II),
                   (d_3 - 2.5 * m_n_34, d_II), (d_4 - 2.5 * m_n_34, d_III)]:
        if df < di:
            error_sum += (10000000 * abs((di - df)) ** 2)


    print("--------------------------------")
    print(f"z1: {z1}, i_12: {i_12:.2f}, i_ges: {i_ges:.2f}, z3: {z3}, beta_12: {beta_12:.2f}, beta_34: {beta_34:.2f}, m_n_12: {m_n_12:.2f}, m_n_34: {m_n_34:.2f}, psi_d_12: {psi_d_12:.2f}, psi_d_34: {psi_d_34:.2f}, sigma_Flim: {sigma_Flim:.2f}")
    print(f"achsabstände: {abs(a_12 - a_34):.2f}")
    return error_sum

# Validierungsfunktion
def validate_solution(z1, i_12, i_ges, z3, beta_12, beta_34, m_n_12, m_n_34, psi_d_12, psi_d_34, sigma_Flim):
    
    z2 = math.ceil(z1 * i_12)

    i_34 = i_ges / i_12

    z4 = math.ceil(z3 * i_34)

 
    print(f"\ni_12: {i_12}, i_34: {i_34}, i_ges: {i_ges}")
    print(f"ggt_12: {np.gcd(z1, z2)}, ggt_34: {np.gcd(z3, z4)}")

    M_t_I = P_I * 1000 / (2 * np.pi * (n_I / 60)) # M_t_I
    M_t_II = M_t_I * i_12 # M_t_II
    M_t_III = M_t_I * i_ges # M_t_III

    print(f"M_t_I: {M_t_I}, M_t_II: {M_t_II}, M_t_III: {M_t_III}")

    M_t_1_eq = M_t_I * K_A
    M_t_3_eq = M_t_II * K_A 

    # Berechne überschlägiges Modul m''
    m_n_min_12 = calc_m_n_min(M_t_1_eq, z1, psi_d_12, beta_12, sigma_Flim)
    m_n_min_34 = calc_m_n_min(M_t_3_eq, z3, psi_d_34, beta_34, sigma_Flim)

    print(f"m_n_min_12: {m_n_min_12}, m_n_min_34: {m_n_min_34}")

 

    print(f"m_n_12: {m_n_12}, m_n_34: {m_n_34}")
    
    M_b_I = M_t_I
    M_b_II = 2 * M_t_II
    M_b_III = M_t_III

    d_I = berechne_welle(M_b_I, M_t_I)
    d_II = berechne_welle(M_b_II, M_t_II)
    d_III = berechne_welle(M_b_III, M_t_III)

    print(f"d_I: {d_I}, d_II: {d_II}, d_III: {d_III}")

    d_1, d_2 = calc_d_z(z1, m_n_12, beta_12), calc_d_z(z2, m_n_12, beta_12)
    d_3, d_4 = calc_d_z(z3, m_n_34, beta_34), calc_d_z(z4, m_n_34, beta_34)

    print(f"d_1: {d_1}, d_2: {d_2}, d_3: {d_3}, d_4: {d_4}")

    a_12, a_34 = (d_1 + d_2) / 2, (d_3 + d_4) / 2
    print(f"a_12: {a_12}, a_34: {a_34}")

    fehler = []
    bestanden = []

    if i_ges_soll * (1 - t_iges) <= i_ges <= i_ges_soll * (1 + t_iges):
        bestanden.append("Übersetzung im Toleranzbereich")
    else:
        fehler.append("Übersetzung nicht im Toleranzbereich")

    if m_n_min_12 <= m_n_12:
        bestanden.append("m_n_12 ausreichend")
    else:
        fehler.append("m_n_12 zu klein")

    if m_n_min_34 <= m_n_34:
        bestanden.append("m_n_34 ausreichend")
    else:
        fehler.append("m_n_34 zu klein")

    if np.isclose(a_12, a_34, atol=1e-3):
        bestanden.append("Achsabstände stimmen")
    else:
        fehler.append("Achsabstände stimmen nicht")

    s_a_12 = 2 +3*m_n_12 
    s_a_34 = 2 +3*m_n_34

    d_k_1 = d_1 + 2 * m_n_12
    if d_k_1/2 + 2.5*s_a_12 + t_b >  h:
        fehler.append(f"Zahnrad passt nicht in Kasten: d_k_1/2 + 2.5*s_a_12 + t_b > h\n Der Unterschied beträgt: {d_k_1/2 + 2.5*s_a_12 + t_b - h:.2f} mm")
    else:
        bestanden.append("Zahnrad passt in Kasten: d_k_1/2 + 2.5*s_a_12 + t_b <= h")   

    d_k_4 = d_4 + 2 * m_n_34
    if d_k_4/2 + 2.5*s_a_34 + t_b >  h:
        fehler.append(f"Zahnrad passt nicht in Kasten: d_k_4/2 + 2.5*s_a_34 + t_b > h\n Der Unterschied beträgt: {d_k_4/2 + 2.5*s_a_34 + t_b - h:.2f} mm")
    else:
        bestanden.append("Zahnrad passt in Kasten: d_k_4/2 + 2.5*s_a_34 + t_b <= h")

    # Prüfe ob Zahnrand 1 auf Welle I passt
    if d_1 - 2.5 * m_n_12 >= d_I:
        bestanden.append("d_f_1 >= d_I")
    else:
        fehler.append("d_f_1 < d_I")

    # Prüfe ob Zahnrand 2 auf Welle II passt
    if d_2 - 2.5 * m_n_12 >= d_II:
        bestanden.append("d_f_2 >= d_II")
    else:
        fehler.append("d_f_2 < d_II")

    # Prüfe ob Zahnrand 3 auf Welle II passt
    if d_3 - 2.5 * m_n_34 >= d_II:
        bestanden.append("d_f_3 >= d_II")
    else:
        fehler.append("d_f_3 < d_III")

    # Prüfe ob Zahnrand 4 auf Welle III passt
    if d_4 - 2.5 * m_n_34 >= d_III:
        bestanden.append("d_f_4 >= d_III")
    else:
        fehler.append("d_f_4 < d_III")

    print("\n✅ Erfüllte Bedingungen:")
    for b in bestanden:
        print(" +", b)

    if fehler:
        print("\n❌ Fehlerhafte Bedingungen:")
        for f in fehler:
            print(" -", f)
    else:
        print("\nAlle Bedingungen erfüllt ✅🎉")

# Hauptausführung
if __name__ == "__main__":
    # Bestimme die Anzahl der CPU-Kerne
    num_cores = multiprocessing.cpu_count()
    print(f"Verfügbare CPU-Kerne: {num_cores}")
    
    
    result = differential_evolution(
        objective, 
        bounds, 
        integrality=intefrality,
        popsize=100, 
        maxiter=1000, 
        seed=42,
        workers=num_cores,  # Verwende alle verfügbaren Kerne
        updating='deferred',  # Verwende sequentielle Aktualisierung
    )

   
   
    print(f"\nMinimaler Funktionswert: {result.fun}")

    x = result.x
    z1 = int(x[0])
    i_12 = x[1]
    i_ges = x[2]
    z3 = int(x[3])
    beta_12 = x[4]
    beta_34 = allowed_beta_34[int(x[5])]
    m_n_12 = allowed_mn[int(x[6])]
    m_n_34 = allowed_mn[int(x[7])]
    psi_d_12, psi_d_34, sigma_Flim = x[8:]

    print("\nOptimale Parameter:")
    print(f"z1={z1}, z2={math.ceil(z1 * i_12)}, z3={z3}, z4={math.ceil(z3 * i_ges / i_12)}")
    print(f"beta_12={beta_12}, beta_34={beta_34}")
    print(f"psi_d_12={psi_d_12}, psi_d_34={psi_d_34}")
    print(f"sigma_Flim={sigma_Flim}")

    validate_solution(z1, i_12, i_ges, z3, beta_12, beta_34, m_n_12, m_n_34, psi_d_12, psi_d_34, sigma_Flim)
    
