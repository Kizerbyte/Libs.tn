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
metingcount = 3

X_as = "T"  # Titel van de kolom in excel zonder nummering
Y_as = "U_h"

X_as_label = r"$t$ [s]"
Y_as_label = r"$U$ [V]"  # de $$ maakt het een math environment & cursief

p0 = [30, 4, 20]  # a0, a1, a2, etc.

# De functieverwerking werkt met np. en math., tot 24 parameters.
# Heb ook ln, log2 en log10 toegevoegd
fitfunctie = "a0 *(1 - np.exp(-(x+a1)/a2))-9.09"

# Kleurcollectie voor iedere dataset
c = ["b", "orange", "r", "g", "black", "purple", "brown", "pink"]

"""
##### Extract en plot data recursief
###############################################################################
###############################################################################
"""

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))  # Creëer figuuromgeving

popt = np.zeros((metingcount, len(p0)))  # Pre-allocation voor referenties
perr = np.zeros((metingcount, len(p0)))

for i in range(metingcount):
    Xas = f"{X_as}{i+1}"  # Automatische nummertoevoeging en behoud X_as
    Yas = f"{Y_as}{i+1}"

    # Data-extractie
    columns = tn.lees_bestand(  # Selecteer kolommen en maak een grote dict.
        Excel_bestand,
        Sheet,
        [Xas, Yas],
    )

    Xas = columns[Xas][1:]  # Tijdswaarden zonder 1e waarde
    Yas = columns[Yas][1:]  # Spanningswaarden van waterpeil

    # Meting is automatisch getimed -> onzekerheid van de .time() functie
    err_Xas = np.ones(len(Xas)) * 16e-3
    # Meting is gedaan met de multimeter
    err_Yas = tn.MMTTi_1604_error(Yas, "U")

    # Plotten van data
    [popt[i], perr[i], _] = tn.sigmaPolynoomfit(
        ax1,
        2,
        Xas,
        Yas,
        err_Xas,
        err_Yas,
        label=r"%s %d" % (Label, i + 1),
        func=fitfunctie,
        p0=p0,
        colour=c[i],
        full_label=False,
    )


"""
##### Plot data
###############################################################################
###############################################################################
"""

ax1.set_title(r"Voorbeeldtitel $U_{h}$ (gain) met tijd")

tn.grafiek_opmaak(
    ax1,
    X_as_label,
    Y_as_label,
    5,
    # xlim=[0, 55],
    # ylim=[-10, 12.5],  # Dit is voor de zoom-in
)

plt.show()  # Niet te vergeten
