from evolution import Individual, Genome
from evolution.genes import CategoricalGene, ValueGene
from evolution.selection import roulette_selection, crossover
import numpy as np
import math

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

#Hilfsfunktion für Drehmoment & Wellendurchmesser
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

class Individual(Individual):
    def __init__(self, genome: Genome | None = None):
        if genome is None:
            super().__init__(Genome(
                [
                ValueGene("beta_12", float, 12.5, (7.5, 17.5)),
                CategoricalGene("beta_34", float, 12.5, [12.5, 15.5]),
                CategoricalGene("m_n_12", float, 2.5, allowed_mn, True),
                CategoricalGene("m_n_34", float, 2.5, allowed_mn, True),
                ValueGene("z1", int, 17, (17, 200)),
                ValueGene("z3", int, 17, (17, 200)),
                ValueGene("i_12", float, 5.5, (5.25,6.0)),
                ValueGene("i_ges", float, 22.5, (i_ges_soll * (1 - t_iges), i_ges_soll * (1 + t_iges))),
                ValueGene("psi_d_12", float, 0.5, (0.35, 0.75)),
                ValueGene("psi_d_34", float, 0.5, (0.35, 0.75)),
                ValueGene("sigma_Flim", float, 310.0, (310.0, 525.0)),
            ]
                ))
        else:
            super().__init__(genome)

    def fitness(self):

        error_sum = 0
        print(f"m_n_12: {self.genome.m_n_12}, m_n_34: {self.genome.m_n_34}")
        
        
        z2 = math.ceil(self.genome.z1 * self.genome.i_12)

        i_34 = self.genome.i_ges / self.genome.i_12

        z4 = math.ceil(self.genome.z3 * i_34)


        error_sum += abs(np.gcd(self.genome.z1, z2) + np.gcd(self.genome.z3, z4))/100


        M_t_I = P_I * 1000 / (2 * np.pi * (n_I / 60)) # M_t_I
        M_t_II = M_t_I * self.genome.i_12 # M_t_II
        M_t_III = M_t_I * self.genome.i_ges # M_t_III

        M_t_1_eq = M_t_I * K_A
        M_t_3_eq = M_t_II * K_A 


        M_b_I = M_t_I
        M_b_II = 2 * M_t_II
        M_b_III = M_t_III

        d_I = berechne_welle(M_b_I, M_t_I)
        d_II = berechne_welle(M_b_II, M_t_II)
        d_III = berechne_welle(M_b_III, M_t_III)
        

        # Berechne überschlägiges Modul m''
        m_n_min_12 = calc_m_n_min(M_t_1_eq, self.genome.z1, self.genome.psi_d_12, self.genome.beta_12, self.genome.sigma_Flim)
        if self.genome.m_n_12 < m_n_min_12:
            error_sum +=  (10000000* abs(self.genome.m_n_12 - m_n_min_12)) **3
            


        d_1 = calc_d_z(self.genome.z1, self.genome.m_n_12, self.genome.beta_12)
        d_2 = calc_d_z(z2, self.genome.m_n_12, self.genome.beta_12)

        m_n_min_34 = calc_m_n_min(M_t_3_eq, self.genome.z3, self.genome.psi_d_34, self.genome.beta_34, self.genome.sigma_Flim)

        
        if self.genome.m_n_34 < m_n_min_34:
            error_sum +=  (100000000* abs(self.genome.m_n_34 - m_n_min_34)) **3
        
        d_3 = calc_d_z(self.genome.z3, self.genome.m_n_34, self.genome.beta_34)
        d_4 = calc_d_z(z4, self.genome.m_n_34, self.genome.beta_34)

    
        
    
        

        a_12 = (d_1 + d_2) / 2
        a_34 = (d_3 + d_4) / 2

        error_sum +=  (10000000* abs(a_12 - a_34)) **4

        d_k_1 = d_1 + 2 * self.genome.m_n_12
        d_k_4 = d_4 + 2 * self.genome.m_n_34

        s_a_1 = 2 +3*self.genome.m_n_12 
        s_a_4 = 2 +3*self.genome.m_n_34

        if d_k_1/2 * 2.5*s_a_1 + t_b > h:
            error_sum += (10000000 * abs(h - (d_k_1/2 * 2.5*s_a_1 + t_b))) ** 2
        if d_k_4/2 * 2.5*s_a_4 + t_b > h:
            error_sum += (10000000 * abs(h - (d_k_4/2 * 2.5*s_a_4 + t_b))) ** 2


        for df, di in [(d_1 - 2.5 * self.genome.m_n_12, d_I), (d_2 - 2.5 * self.genome.m_n_12, d_II),
                    (d_3 - 2.5 * self.genome.m_n_34, d_II), (d_4 - 2.5 * self.genome.m_n_34, d_III)]:
            if df < di:
                error_sum += (10000000 * abs((di - df)) ** 2)


        print("--------------------------------")
        print(f"z1: {self.genome.z1}, i_12: {self.genome.i_12:.2f}, i_ges: {self.genome.i_ges:.2f}, z3: {self.genome.z3}, beta_12: {self.genome.beta_12:.2f}, beta_34: {self.genome.beta_34:.2f}, m_n_12: {self.genome.m_n_12:.2f}, m_n_34: {self.genome.m_n_34:.2f}, psi_d_12: {self.genome.psi_d_12:.2f}, psi_d_34: {self.genome.psi_d_34:.2f}, sigma_Flim: {self.genome.sigma_Flim:.2f}")
        print(f"achsabstände: {abs(a_12 - a_34):.2f}")
        return error_sum

def main():

    population = [Individual() for _ in range(10)]

    for generation in range(100):
        population = sorted(population, key=lambda x: x.fitness())
        print(f"Generation {generation} beste Fitness: {population[0].fitness()}")

        parents = roulette_selection(population, 90)
        children = crossover(parents)

        population = population[:10] + children

    print(population[0])

if __name__ == "__main__":
    main()