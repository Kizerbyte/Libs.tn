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
Excel_bestand = "LibsTNSinusdata.xlsx"  # Selecteer bestand
Sheet = "Sheet1"  # Selecteer blad


Label = "Hall effect sensor"

Xas = "T"  # Titel van de kolom in excel
Yas = "A0"  # Dataset bevat A0 en A1, probeer maar!

Xas_label = r"$t$ [ms]"
Yas_label = r"$U$ [mV]"  # de $$ maakt het een math environment & cursief


p0 = [200, 150, 1, 2000]  # a0=A; a1=periode; a2=fase; a3=translatie

# De functieverwerking werkt met np. en math., tot 24 parameters.
# Heb ook ln, log2 en log10 toegevoegd
# Algemene vorm van sinusfunctie:
stringfunc = "a0 * np.sin(2 * np.pi / a1 * x + 2 * np.pi * a2) + a3"


startpunt = 2800  # Datapunt, niet tijdstip
eindpunt = 3000


"""
##### Extract data
###############################################################################
###############################################################################
"""

columns = tn.lees_bestand(  # Selecteer kolommen en maak een grote dict.
    Excel_bestand,
    Sheet,
    [Xas, Yas],  # Plaats hier de kolommen die je wilt
)

Xas = columns[Xas][startpunt:eindpunt]  # Dict->array en vervang Xas variabele
Xas = Xas - Xas[0]  # het beginpunt op t=0 zetten
err_Xas = np.ones(len(Xas)) * 1e-3  # 1ms onzekerheid van ticks_ms()

Yas = columns[Yas][startpunt:eindpunt]
err_Yas = np.ones(len(Yas)) * 0.8 / 2
# 3.3V/4095 = 0.8 mV resolutie van de RPi Pico


"""
##### Plot data
###############################################################################
###############################################################################
"""

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))  # Creëer figuur met formaat

tn.sigmaPolynoomfit(
    ax1,  # Kies de grafiek
    2,  # Kies het model
    Xas,
    Yas,
    err_Xas,
    err_Yas,
    label=Label,
    func=stringfunc,  # Zie definitie hierboven
    p0=p0,
    full_label=False,  # Dit houdt de legenda minimaal
)

tn.grafiek_opmaak(
    ax1,  # Kies de grafiek
    Xas_label,
    Yas_label,
    5,  # Legenda locatie 5 is extra
    # xlim=[0, 0],
    # ylim=[0, 0]   # Dit is voor een zoom-in handig
    ncol=1,  # 2 kolommen was te breed ivm de formule
)

plt.show()  # Niet te vergeten
