"""Created on Fri Oct 11 14:27:41 2024 by Floris Messack."""

import matplotlib.pyplot as plt
import numpy as np

from Libs import tn

plt.close("all")  # Kan je weghalen, kan je laten, vrije keuze.

"""
##### Variabelen
###############################################################################
###############################################################################
"""
# Dit bestand voegt automatisch de meetonzekerheid van de MMTTi-1604 toe.
# Alleen meetdata is benodigd, van twee assen.
Excel_bestand = "LibsTNvoorbeelddata.xlsx"  # Selecteer bestand
Sheet = "Sheet1"  # Selecteer blad


Label = "Stapresponsie"

Xas = "T1"  # Titel van de kolom in excel
Yas = "U_h1"

Xas_label = r"$t$ [s]"
Yas_label = r"$U$ [V]"  # de $$ maakt het een math environment & cursief


p0 = [30, 4, 20]  # a0, a1, a2, etc.

# De functieverwerking werkt met np. en math., tot 24 parameters.
# Heb ook ln, log2 en log10 toegevoegd
stringfunc = "a0 *(1 - np.exp(-(x+a1)/a2))-9.09"
# stringfunc = "a0 - a1 * x ** a2"  # Deze fit faalt, kijk maar

"""
##### Extract data
###############################################################################
###############################################################################
"""

columns = tn.lees_bestand(  # Selecteer kolommen en maak een grote dict.
    Excel_bestand,
    Sheet,
    [Xas, Yas],  # Plaats hier de kolommen die je wilt!
)

Xas = columns[Xas]  # Haal waarden uit dict en vervang Xas string
# Hier kan je spelen met de onzekerheid, voor het effect
err_Xas = np.ones(len(Xas)) * 16000e-3  # 16ms onzekerheid (van .time())

Yas = columns[Yas]
err_Yas = tn.MMTTi_1604_error(Yas, "U")  # Fout van de multimeter


"""
##### Plot data
###############################################################################
###############################################################################
"""

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))  # Creëer figuur met formaat


tn.sigmaPolynoomfit(
    ax1,  # Kies de grafiek
    2,  # Kies het model. Merk op hoe methode 1 geen x-as onzekerheid meeneemt
    Xas,
    Yas,
    err_Xas,  # De fout vd meting (zelfde eenheid als de meting)
    err_Yas,
    label=Label,
    func=stringfunc,  # Zie definitie hierboven
    p0=p0,
)


tn.grafiek_opmaak(
    ax1,  # Kies de grafiek
    Xas_label,
    Yas_label,
    5,  # Legenda locatie 5 is extra
    # xlim=[0, 0],
    # ylim=[0, 0]   # Dit is voor de zoom-in handig
)

plt.show()  # Niet te vergeten
