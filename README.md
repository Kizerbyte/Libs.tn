# Libs.tn
Automatische data extractie en dataverwerking voor de TN studie aan de HHS.

Inclusief MMTTi 1604 meetfoutverwerking en afronding o.b.v. onzekerheid.


Uitleg om het werkend te krijgen.
1. Via de zoekfunctie op je computer, open Anaconda Prompt.
2. Plak het volgende in de terminal:
```console
pip install numpy matplotlib pandas sympy scipy
```
3. Plaats de map Libs in je python working environment (de folder rechtsbovenin Spyder aangegeven)
4. Om de code toe te passen, zie de bijgeleverde voorbeeldbestanden.

Als je telkens in een andere working environment zit, kan je m vast zetten via:
Tools->Preferences->Working directory en dan de folder daar te selecteren.
Dat is dan je standaard python folder. (Of neem het mapje telkens mee? Doe wat je wilt.)

Ik heb geprobeerd de documentatie zo compleet mogelijk te maken, waarbij de netheid met de jaren is toegenomen.
Sommige stukken code komen van stackoverflow dus commentaar is engels.
Veel soorten errors worden opgevangen en aan je uitgelegd, tenzij de errormsg zelf voldoende duidelijk is.

Als je wilt weten welke functies er in het bestand zitten, run tn.py voor een lijst of 
```console
help(tn.<functie>)
```
