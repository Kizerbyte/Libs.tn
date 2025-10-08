import pyvisa  # het uitlezen/aansturen van meetinstrumenten
import time  # voor het tijdsinterval van elke loop
import sys  # voor het stoppen van het script na zero'en
import threading  # voor de cancel-interrupt
import numpy as np
from pynput import keyboard  # het aansturen van het keyboard
from ThorlabsPM100 import ThorlabsPM100  # De PyVISA wrapper voor de pm

"""
##### Installatie ############################################################
"""

# https://pythonhosted.org/ThorlabsPM100/ voor documentatie

# Maakt gebruik van Thorlabs PM100 wrapper voor PyVISA
"""pip instal ThorlabsPM100, pyvisa"""
# PyVISA is een wrapper voor NI-VISA (een standaard voor meetinstrumenten
# uitlezen) Vrij groot programma, maar is wel universeel voor alle instrumenten
# https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html#570633

"""Belangrijk!"""
# Wanneer je je PM100D aangesloten hebt en NI-VISA geinstalleerd hebt, open
# device manager en update de driver van de PM100D. Klik daarna op
# 'bestanden op deze computer' en 'kies zelf een bestand uit een lijst'
# en update naar de USBTMC driver.

"""
##### Functies powermeter ####################################################
"""


def pm_autoconnect():
    """Gebruikt de vendor identifier om een ThorLabs apparaat te selecteren.

    Returns
    -------
    power_meter : object
    """
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()  # "USB0::0x0000::0x0000::xxxxxxxxx::INSTR"
    match = [r for r in resources if "USB0::0x1313" in r and "INSTR" in r]
    if not match:  # Elke vendor heeft een eigen identifier
        raise RuntimeError(f"No ThorLabs identifier found in list {resources}")

    inst = rm.open_resource(match[0], timeout=1)
    try:
        inst.term_chars = "\n"  # "Alleen nodig bij NI-VISA, niet bij USBTMC
    except Exception:
        pass
    power_meter = ThorlabsPM100(inst=inst)
    return power_meter


def configure_pm_for_power(pm, wavelength_nm, interval):
    """Veel settings zijn per hardware sessie, zoals averaging."""
    pm.configure.scalar.power()  # Turn to power measurement
    pm.sense.power.dc.unit = "W"  # Measurement unit
    pm.sense.correction.wavelength = wavelength_nm  # Te meten wavelength
    pm.sense.power.dc.range.auto = "ON"
    # sample averaging on last N samples. Based on <80% measurement interval
    pm.sense.average.count = max(1, np.floor(interval * 10 * 0.8))
    pm.sense.average.state = True


def zero_adjustment(pm, delay=3):
    """Gebruik voor metingen bij lage lichtintensiteit.

    Trekt de huidige achtergrondruis van de data af.
    Reset automatisch na elke hardware sessie
    """
    print("Oude nulwaarde:", pm.sense.correction.collect.zero.magnitude, "W")
    print("Sensorkalibratie in {delay/2} seconden")
    time.sleep(delay / 2)
    pm.sense.correction.collect.zero.initiate()
    time.sleep(delay / 2)
    print("Nieuwe nulwaarde:", pm.sense.correction.collect.zero.magnitude, "W")


"""
##### Loop-exit functies #####################################################
"""
_cancel = threading.Event()


def _on_key_press(key):
    if key == keyboard.Key.esc:  # Gebruik ESC om de meting te stoppen
        _cancel.set()
        return False


listener = keyboard.Listener(on_press=_on_key_press)
listener.start()


"""
##### Parameters #############################################################
"""


n_steps = 50  # Aantal meetpunten in een sweep
interval = 1  # Tijdsinterval tussen metingen
# PM100D output is ~10Hz via USB, dichtbij die waarde geeft duplicate readings.

start_delay = 5
wavelength_nm = 650  # Kalibratiegolflengte voor de fotodiode

nulmeting = False


"""
##### Script #################################################################
"""


def main():
    """Docstring."""
    PM100D = pm_autoconnect()
    configure_pm_for_power(PM100D, wavelength_nm, interval)

    if nulmeting:  # Nulmeting aan/uit
        zero_adjustment(PM100D)  # Kalibreert de nul obv strooilicht
        sys.exit(0)

    """
    ######################################################
    ############# Hieronder je preallocaties #############
    """

    times = np.zeros(n_steps + 1)  # Preallocation
    values = np.zeros(n_steps + 1)

    print(f"Script start in {start_delay} seconden...")
    time.sleep(start_delay)

    """
    ############# Hierboven je preallocaties #############
    ######################################################
    """
    t0 = time.perf_counter()  # Heeft veel hogere nauwkeurigheid dan .time()
    for i in range(n_steps + 1):  # meting 0 én n_steps
        if _cancel.is_set():  # Catch exit
            print("Gestopt met ESC")
            sys.exit(0)
        """
        ########################################################
        ########## Hieronder je uitvoering in de loop ##########
        """

        values[i] = PM100D.read
        times[i] = time.perf_counter() - t0

        print(f"t={times[i]:6.3f}s | P={values[i]:.6e} W")

        """
        ########## Hierboven je uitvoering in de loop ##########
        ########################################################
        """
        time.sleep(interval)  # Timeout interval (Na het veranderen vd input)

    return times, values


if __name__ == "__main__":
    try:
        a_tijd, a_Meting = main()
    except KeyboardInterrupt:
        print("Gestopt met Ctrl+C")  # Houdt het netjes
