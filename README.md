# Libs.tn
Automatische Excel data extractie en dataverwerking voor de TN studie aan de HHS Delft. De code is samengekomen met de jaren en telkens verbeterd.

Redenen waarom dit een chille module is
* MMTTi 1604 meetfoutverwerking
* Afronding o.b.v. onzekerheid (in r-string vorm)
* Super simpele opmaak van grafieken
* Data fitten door middel van een string (e.g. "a0 * x + a1") met een praktisch onbeperkt aantal parameters
* Onzekerheidsinterval rondom de fitfunctie ongeacht de functievorm (ook sinusvorm).
* 3 soorten curvefits (Nonlinear Least Squares, NLS met Monte Carlo methode, Orthogonal Distance Regression)
* Snelle workflow door gegeneraliseerde code en mogelijkheid tot bijna-live databewerking in Excel.
* Prima geoptimaliseerd voor snelheid.
* De code mansplaint vrijwel altijd wat voor een fout je hebt gemaakt. Oorzaak achterhalen via de documentatie is makkelijk.

Uitleg om het werkend te krijgen.
1. Via de zoekfunctie op je computer, open Anaconda Prompt.
2. Plak het volgende in de terminal:
```console
pip install numpy matplotlib pandas sympy scipy
```
3. Plaats de map Libs in je Python working environment (de folder rechtsbovenin Spyder aangegeven)
4. Om de code toe te passen, zie de bijgeleverde voorbeeldbestanden.

Als je telkens in een andere working environment zit, kan je m vast zetten via:
Tools->Preferences->Working directory en dan de folder daar te selecteren.
Dat is dan je standaard Python folder. (Of neem het mapje telkens mee, een vrije keuze)

Ik heb geprobeerd de documentatie zo compleet mogelijk te maken, waarbij de netheid met de jaren is toegenomen.
Sommige stukken code komen van bijv. Stack Overflow dus dat commentaar is Engels.
Veel soorten errors worden opgevangen en aan je uitgelegd, met een debug tree zodat je kan achterhalen waar de fout vandaan komt.

Lees de documentatie van sigmaPolynoomfit() goed door! Dan weet je wat 'ie kan.

Als je wilt weten welke functies er in het bestand zitten run tn.py, of per functie specifiek doe
```console
help(tn.<functie>)
```
P.S.

Voor het wisselen van punten en comma's zoals het hoort in Nederlandse notatie, gebruik de TIS-TN module op https://github.com/ddland/TIS-TN-python-code.
Je kan dan spelen met de tn.Reglabelmaker() functie en het aanpassen.

Ook is de aansturing van de MMTTi 1604 erg nuttig, zie https://github.com/ddland/PythonCode/tree/main/tti1604
