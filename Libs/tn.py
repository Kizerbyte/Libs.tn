"""
Floris Messack, HHS : 2023.

Kleine module met functies die mij teveel tijd hebben gekost, maar ook veel
tijd hebben bespaard tijdens data-analyse.
Hoe het te gebruiken? Zie de voorbeeldbestanden die meegegeven zijn.
Download : https://github.com/Kizerbyte/Libs.tn
S.O. naar J. van Tol voor de openbaring van het onzekerheidsinterval.
Zie: uncertainty_range()
"""

import math
import re as re2  # naming overlap (extracting_variables functie)
import string  # extracting_variables
import sys  # Voor de abort functie bij een error
import warnings  # Voor error handling van de ODR fit
from pathlib import Path  # voor vinden van downloads folder


import numpy as np
import pandas as pd  # Voor het inlezen van de excel
from scipy.optimize import curve_fit, OptimizeWarning
from sympy import diff  # Voor het onzekerheidsgebied van de functie
from sympy import lambdify
from sympy import Matrix
from sympy import re
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from sympy import symbols
from sympy import sympify


class ManualError(Exception):
    """Handmatige class voor het oproepen van fouten.

    Hierdoor kan je de tree zien en zelf debuggen door te klikken op de groene
    filepaths in de error
    """

    def __init__(self, *args):
        message = " ".join(str(arg) for arg in args)
        super().__init__(message)


""" ##############################################
                                                        Importeer module met:
                                                        from Libs import tn
############################################## """

# %% Data extractie
"""
#################
#######################    Data Extractie    #################################
#########
"""


def lees_bestand(file, sheet="Sheet1", cols=["U", "I"], debug=False):
    """
    Kolommen uit een excelsheet halen en omzetten in een werkbare dictionary.

    Parameters
    ----------
    file : str
        Bestandsnaam die zich op hetzelfde C:/ pad bevindt
        óf in C:/User/<username>/Downloads/
    sheet : str
        Default is 'Sheet1'
    cols : str, list
        Één of meer kolomtitels (Default = ['U', 'I'])
    debug : bool
        Zet True als je de volledige pandas dataframe wilt van de excel

    Returns
    -------
    Kolommen : dict
        Opgevraagde kolommen als één numpy dictionary
    Excel : df (wanneer debug=True)
        De volledige pandas dataframe van de excel en de NaN-count per kolom
    """

    def vind_bestand(path, file):  # Return dataframe alleen indien gevonden
        p = Path(path / file)
        if p.is_file():
            Excel = pd.read_excel(p, sheet)
            print(f"\nBestand gevonden! {path}\\{file} ")
            print(
                "Zoekend naar kolom"
                + ("men" if len(cols) > 1 else "")  # Is mooi
                + f" {cols} in '{sheet}'...",
                end="",
            )
            if not debug:
                return zoek_kolom(Excel, cols)
            else:
                return zoek_kolom(Excel, cols, debug=True), Excel

    pathlist = [
        Path(sys.argv[0]).resolve().parent,  # Folder van script dat m oproept
        Path.home() / "Downloads",  # Donwloadsfolder
        Path.cwd(),  # Working directory (zie de path rechtsbovenin in Spyder)
    ]

    for pad in pathlist:
        Excel_df = vind_bestand(pad, file)
        if Excel_df is not None:
            return Excel_df

    # Gaat verder bij falen van pathfinding
    print(
        "\nBestand niet gevonden in scriptfolder, downloadsfolder of working",
        f"directory.\nWorking directory: {Path.cwd()}\nAborted script.",
    )
    sys.exit()


def zoek_kolom(dataframe, kolommen, debug=False):
    """
    Kolommen vinden in een panda's df en vertalen naar Numpy, met errorfunctie.

    Parameters
    ----------
    dataframe : df
        De dataframe waarin gezocht moet worden
    kolommen : str
        kolomtitel(s) die gezocht moet(en) worden
    debug : bool
        Zet True als je de wilt weten hoeveel NaNs

    Returns
    -------
    newdict : dict
        met specifiek die kolommen
    print (wanneer debug=True)
        NaN count
    """
    newdict = {}
    for col in kolommen:
        try:
            # Zoek de kolom met metingen en creëer een dict entry met deze data
            newdict.update({col: dataframe[col].dropna().to_numpy()})
            if debug and (dataframe[col].isnull().values.sum() != 0):
                print(
                    "\n-debug- Kolom '{}' bevat {} NaN(s)".format(
                        col,
                        dataframe[col].isnull().values.sum(),
                    ),
                    end="",
                )
        except KeyError:
            print(f"\n\nKolom '{col}' niet gevonden!")
            sys.exit()
    print("\u2705")  # checkmark
    return newdict


# %% Opmaak
"""
#################
#######################        Opmaak        #################################
#########
"""


def sci_error(number, sig_number, sci_bucket=[4, -2]):
    """Voor het omzetten van 5E+2 notatie naar 5*10^2 dmv de error.

    Het maakt gebruik van de significantie van de error, welke omghoog
    afgerond wordt naar 1 digit. Het wordt teruggegeven als één r-string.
    NB: Het is nodig het nog toe te voegen tussen r'$' + return + '$'.

    Voorbeeld:
    import matplotlib.pyplot as plt
    from Libs import tn  # Eigen library
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 1))  # Creeer figuur
    ax1.set_title(r"$" + tn.sci_error(0.003, 0.00062) + "$")

    Parameters
    ----------
        number : int, float
            De meetwaarde.
        sig_number : int, float
            De fout van de meetwaarde.
        sci_bucket : array
            De max en min ordegrootte waartussen geen wetenschappelijke notatie
            is.

    Returns
    -------
        value : r-str
            De meetwaarde en onzekerheid met de correct vervoegde onzekerheid.
    """
    if number != 0 and sig_number != 0:  # Normaal geval
        try:
            n = 0
            numpow = math.floor(np.log10(abs(number)))
            n = 1
            sigpow = math.floor(np.log10(abs(sig_number)))
        except OverflowError:
            raise ManualError(
                "Afrondingsfout: Er is 'inf' gegeven als",
                ("meetwaarde" if n == 0 else "onzekerheid"),
                "bij afronding. Check de inputdata voor deze functie. \n",
            ) from None
    elif number == 0:  # Bij een nul moet ie niet kutten
        return "0"
    else:
        return "%s" % number  # Bij een onzekerheid van nul is het gegeven

    power = numpow - sigpow  # the difference of orders
    if power < 0:
        raise ManualError(
            f"Afrondingsfout: Meetwaarde {number:.3e} is kleiner dan",
            f"onzekerheid {sig_number:.3e} ! Check p0",
        )
    ret_string = "{0:.{1:d}e}".format(number, power)
    a, b = ret_string.split("e")

    # remove leading "+" and strip leading zeros
    b = int(b)

    # Roundup the error to the last digit as int
    sigval = math.ceil(sig_number / 10**sigpow)

    # Decide if decimal places or integer
    if sci_bucket[1] <= numpow <= sci_bucket[0] and sigpow <= 0:
        e = sigval * 10**sigpow  # Lift to original power
        return f"(%.{-sigpow}f\\pm %.{-sigpow}f)" % (  # noqa
            round(number, -sigpow),
            e,
        )
    else:
        # Lift to order of the 'number'
        e = round(sigval / 10**power, power)
        if power == 0:
            e = int(e)
        return r"(%s\pm %s)\cdot 10^{%d}" % (a, e, b)  # noqa


def Reglabelmaker(formula, popt, perr, full_label=True):
    """Legenda opmaak van regressielijn klaarmaken."""
    legendafunc = extract_variables(
        formula,
        replace_with_values=True,
        popt=[sci_error(p, e) for p, e in zip(popt, perr)],
    )
    # try:
    rstring_formula = (
        re2.sub(
            r"log\(",
            r"ln(",
            re2.sub(
                r"\*\*",
                r"$^$",
                re2.sub(
                    r"\s\*\s",
                    r" \\cdot ",
                    re2.sub(r"log(\d+)\(", r"^{\1}log(", legendafunc),
                ),
            ),
        )
        .replace(r"exp(", r"e$^$(")
        .replace("math.", "")
        .replace("np.", "")
        .replace("\\cdot pi", "\\pi")
        .replace(r" pi", "\\pi")
    )
    if full_label:
        return r"Regressie %s: $y = " + rstring_formula + "$"  # noqa
    else:
        return r"$y = " + rstring_formula + "$"  # noqa


# %% Formula-string handling
"""
#################
#####################  Formula-string handling  ##############################
#########
"""


def parameter_list(formula):
    """Convert formula to parameter list in encountered order.

    Takes in formula in string form and spits out an ordered spaced string.

    Parameters
    ----------
        formula : str
            e.g. "a0 * x ** a1 + a2" (up to 23 parameters)

    Returns
    -------
        list : parameters in encountered order.

    """
    exclude = set(dir(np)).union(dir(math), {"np", "math"})
    return [
        name
        for name in dict.fromkeys(re2.findall(r"\b[a-zA-Z_]\w*\b", formula))
        if name not in exclude
    ]


def extract_variables(
    formula,
    rename=False,
    replace=False,
    replace_with_values=False,
    popt=None,
):
    """Extract or replace parameter names in a formula string.

    Optionally replace them with values.

    Parameters
    ----------
        formula : str
            e.g. "a0 * x ** a1 + a2" (up to 25 parameters)
        rename  : bool
            If True, returns a space-separated string with the parameters
            replaced by a-z.
        replace : bool
            If True, returns the formula with parameters replaced by a-z.
        replace_with_values : bool
            If True, returns the formula with parameters replaced by values
            from popt.
        popt : list or array
            Values to replace the parameters with (only needed if
                                                   replace_with_values=True).

    Returns
    -------
        str :
            If all is False, returns a space-separated string of the
            original parameters.
    """
    parameters = parameter_list(formula)

    # Mapping the parameters to single-letter variables
    mapping = {
        v: string.ascii_lowercase[i]
        for i, v in enumerate([t for t in parameters if t != "x"])
    }
    mapping["x"] = "x"

    # If replace_values is True, substitute the variables with popt[]
    if replace_with_values:
        if popt is None:
            raise ValueError("⚠️ replace_with_values=True vereist waarden")

        # Check if number of values matches number of variables (excluding 'x')
        if len(popt) != len(mapping) - 1:
            raise ValueError("Het aantal parameters komt niet overeen met p0")

        # Map variables to values
        var_value_dict = {
            v: popt[i]
            for i, v in enumerate([t for t in parameters if t != "x"])
        }

        # Replace variables in formula with their corresponding values
        def value_replacer(match):
            var = match.group(0)
            # Return value or leave variable if not in dictionary
            return str(var_value_dict.get(var, var))

        return re2.sub(r"\b[a-zA-Z_]\w*\b", value_replacer, formula)

    if replace:
        # Replace variable names with single-letter variables (a-z)
        def replacer(match):
            # Use mapping for variables
            return mapping.get(match.group(0), match.group(0))

        return re2.sub(r"\b[a-zA-Z_]\w*\b", replacer, formula)
    elif rename:
        # Return mapped letters + 'x'
        return " ".join(mapping[v] for v in mapping if v != "x") + " x"
    else:
        parameters.append(parameters.pop(parameters.index("x")))
        return " ".join(parameters)


def check_for_parameters(p0, formula):
    """Check if ammount of entries in p0 match the parameter count in formula.

    Also checks if functions are .numpy or .math compatible
    Specifically uses a0, a1, a2, ... an.

    Prints log of missing and/or redundant parameters.

    Parameters
    ----------
        p0      : list
        formula : string
            "a0 * x + a1"
    Returns
    -------
        check : bool

    """
    # Build expected set of parameters: a0 ... a(n-1) plus 'x'
    expected = {f"a{i}" for i in range(len(p0))}
    expected.add("x")

    # Exclude known functions/constants from numpy and math
    exclude = set(dir(np)).union(dir(math), {"np", "math"})

    # Keep only the user-defined variables
    used = set(parameter_list(formula)) - exclude

    missing = expected - used
    extra = used - expected

    if missing:
        print("\np0 mist parameter(s)", ", ".join(sorted(missing)))
    if extra:
        print(
            "\nOnbruikbare parameters of niet herkend door numpy/math:\n\t",
            ", ".join(sorted(extra)),
        )

    return not missing and not extra


def set_function_shape(formula):
    """Define the shape of the callable function.

    Uses the given formula to create a callable function of shape
    for use in scipy ODR()

    Parameters
    ----------
        formula : str
            example: "a0 * x + a1"
    Returns
    -------
        ODR=True : lambda function
            of shape func([a0, a1], x)
        OR
        ODR=False : function
            of shape func(x, a0, a1)
    """
    # Extract variable names
    vars = parameter_list(formula)
    vars.remove("x")
    vars.insert(0, "x")  # x als eerste in de rij

    # Eval function calls limiteren aan de safe_globals. Is sneller en veiliger
    safe_globals = {
        "np": np,
        "math": math,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "abs": np.abs,
    }
    # Alle varianten van log_n(x) herschrijven
    formula = re2.sub(r"log(\d+(?:\.\d+)?)\(", r"(1/log(\1))*log(", formula)

    def model(x, *beta):
        x = np.asarray(x, dtype=np.float64)
        # unpack beta if it's a single list/array
        if len(beta) == 1 and isinstance(beta[0], (list, np.ndarray)):
            beta = beta[0]
        local_vars = dict(zip(vars[1:], beta))
        local_vars["x"] = x
        return eval(formula, safe_globals, local_vars)

    return model


# %% Extra dataverwerking
"""
#################
####################### Extra dataverwerking #################################
#########
"""


def plot_confidence_band(
    ax,
    func,
    popt,
    perr,
    x_vals,
    formulastring=None,
    colour="blue",
    N=1000,
):
    """
    Plot een 3σ intervalgebied rondom popt data via Monte Carlo-simulatie.

    Voor elke van de N iteraties worden parameters willekeurig gekozen volgens
    een normale verdeling (rond popt met standaardafwijking perr). Hiermee
    wordt het model geëvalueerd en ontstaat een spreiding van uitkomsten per x-
    waarde.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    func : callable
        de functie, van de vorm hebben van set_function_shape().
    popt : array-like
    perr : array-like
    x_vals : array-like
        De x-waarden waarover de functie geëvalueerd en geplot wordt.
    formulastring : str
        String met de oorspronkelijke formule, voor automatische labelgeneratie
        via Reglabelmaker().
    colour : str
    N : int
        Aantal Monte Carlo-simulaties.
    """
    popt = np.array(popt)
    perr = np.array(perr)
    n_params = len(popt)

    # Generate samples for each parameter
    samples = [np.random.normal(popt[i], perr[i], N) for i in range(n_params)]

    # Evaluate function for each parameter sample
    y_samples = np.zeros((N, len(x_vals)))
    for i in range(N):
        params = [samples[j][i] for j in range(n_params)]
        y_samples[i] = func(*params, x_vals)

    # Calculate percentile bands
    band_3sigma_lower = np.percentile(y_samples, 0.135, axis=0)
    band_3sigma_upper = np.percentile(y_samples, 99.865, axis=0)

    best_fit = func(*popt, x_vals)
    if formulastring is None:
        regressionlabel = "Beste fit"
    else:
        regressionlabel = (
            Reglabelmaker(formulastring, popt, perr) % "beste fit"
        )
    # Plot
    ax.plot(x_vals, best_fit, color=colour, label=regressionlabel)
    ax.fill_between(
        x_vals,
        band_3sigma_lower,
        band_3sigma_upper,
        color=colour,
        alpha=0.2,
        label="3σ (~99.7%)",
    )


def round_up(val):
    """Rondt de waarde omhoog af naar het meest significante getal.

    225 -> 300 etc.

    Parameters
    ----------
        val : int, float
            Waarde om af te ronden
    Returns
    -------
        rounded : int
    """
    b = math.floor(math.log10(val))
    return math.ceil(val / 10**b) * 10**b


def MMTTi_1604_error(meetdata, UI="U"):
    """Functie om de fout in de meting te bepalen van de MMTTi-1604.

    Werkt per waarde of verwerkt dict met 1-2 kolommen

    Geeft meetdata terug met 1 significant getal, zoals onzekerheden horen.

    Parameters
    ----------
        meetdata : int, float, labeled array OR dict
        UI       : str
            Indien 1 meetwaarde gegeven
            Spanning 'U' of stroom 'I'

    Returns
    -------
        data_verwerkt : list OR dict
            Onzekerheid van de meting
    """
    error_array = {  # U en I meetspecificaties van de MMTTi 1604
        "U": {
            "range": [0.400, 4, 40, 400, 1000],
            "accuracy": [0.0008, 0.0008, 0.0008, 0.0008, 0.0009],
            "resolutie": [10e-6, 100e-6, 1e-3, 10e-3, 100e-3],
            "digits": [4, 4, 4, 4, 4],
        },
        "I": {
            "range": [0.004, 0.400, 1, 5, 10],
            "accuracy": [0.001, 0.001, 0.003, 0.010, 0.030],
            "resolutie": [0.1e-6, 10e-6, 1e-3, 1e-3, 1e-3],
            "digits": [4, 4, 4, 4, 10],
        },
    }

    def MMTTi_error(data, UI):
        """Selecteert de meetfout per gegeven waarde."""
        for index, range_max_value in enumerate(error_array[UI]["range"]):
            if (
                data < range_max_value
            ):  # Zodra het kleiner is heeft ie de range
                accuracy_value = error_array[UI]["accuracy"][index]
                resolutie_value = error_array[UI]["resolutie"][index]
                n_o_digits = error_array[UI]["digits"][index]
                break
        try:
            data_error = (
                abs(data) * accuracy_value + n_o_digits * resolutie_value
            )
        except UnboundLocalError:
            raise ManualError(
                "⚠️ %s=%d is buiten het bereik van de MMTTi" % (UI, data),
            )

        data_error_rounded = round_up(data_error)

        return data_error_rounded

    if isinstance(meetdata, int):
        # Standaard werking van functie
        return MMTTi_error(meetdata, UI)

    elif isinstance(meetdata, (np.ndarray, list)):
        # Ingebouwde loop voor arrayverwerking
        meetdata_err = np.zeros(len(meetdata))
        for i, meting in enumerate(meetdata):
            meetdata_err[i] = MMTTi_error(meting, UI)
        return meetdata_err

    elif isinstance(meetdata, dict):
        print("MMTTi took dictionary")
        # Ingebouwde loop voor dictionaryverwerking
        meetdata_err = {}
        for kolom in meetdata:
            print("Found", kolom)
            meetdata_err.update(
                {"err_" + kolom: np.zeros(len(meetdata[kolom]))},
            )
            for i, meting in enumerate(meetdata[kolom]):
                meetdata_err["err_" + kolom][i] = MMTTi_error(meting, kolom)
        return meetdata_err
    else:
        raise ManualError(
            "MMTTi_1604_error() verwacht een int, dict of list.",
            "Ontving %s" % type(meetdata),
        )


# %% Dataverwerking subfuncties
"""
#################
#######################      De fits     #################################
#########
"""


def uncertainty_range(formula, popt, pcov, Xspace):
    """3sigma interval met SymPy, accepteert een variabel aantal parameters(!).

    Stappenplan:
        Haalt de variabelen uit de formule en hernoemt ze a-w (voor SymPy)
        definiëert x als variabele, rest als parameters
        Fixt een aantal standaard syntax-moeilijkheden.
        Neemt de partiëel afgeleide van de functie naar elke parameter in een
        matrix (gradiëntmatrix). Dit laat zien wat de afhankelijkheid is t.o.v.
        elke parameter.

        Voor optimalisatie schrijft het elke part.afg. als een lambda functie.
        Daarna in een forloop van elk punt op de linspace, vult het de x & popt
        in en berekent het numeriek de afgeleide van elke partiëel afgeleide.
        Samen met de covariantiematrix (pcov van de fit) wordt de std.dev
        uitgerekend bij elk punt en vastgelegd in XspaceErr[i]

        Uitkomst is een gewogen onzekerheid afhankelijk van lokale
        meetpuntonzekerheid en meetpuntkwantiteit.
        Dus een puntverdeling als een longitudinale golf, zorgt ook voor een
        dubbel-sinusoïdaal gebied in XspaceErr

    Parameters
    ----------
        formula : string
                    "a0/a1 * (1 - np.exp(-x/a1) + a2" doe eens gek
        popt : list
            Optimale parameters van de voorafgaande fit
        pcov : 2D-list
            Covariantiematrix van de fitparameters
    Returns
    -------
        XspaceErr : list
            Het onzekerheidsinterval van de fit over de linspace
    """
    param_string = extract_variables(formula)  # "a0 a1 x" used for symbols()
    param_keys = param_string.split()  # Create the key names for making dict

    # Alle parameters definiëren als symbolen in SymPy
    symbol_list = symbols(param_string)  # (a0, a1, x) --> sympy-type object
    symbol_dict = dict(zip(param_keys, symbol_list))  # ('a0':a0) voor oproepen

    x_symbol = symbol_dict["x"]  # pointer naar x als sympy-type object

    # Speciale functies zoals sin, exp, log_n() herschrijven voor SymPy
    clean_formula = (
        re2.sub(r"log(\d+(?:\.\d+)?)\(", r"(1/log(\1))*log(", formula)
        .replace("math.", "")
        .replace("np.", "")
    )
    # De formule overhandigen aan SymPy (Symbolic Python)
    sympy_expr = sympify(clean_formula, locals=symbol_dict)

    # Creëer een lijst met elk sympy-symbol behalve die met de 'x'-key
    param_symbols = [symbol_dict[v] for v in param_keys if v != "x"]

    # Gradiënt berekenen (partiëel afleiden i.v.t. elke parameter in de lijst)
    grad = Matrix([diff(sympy_expr, p) for p in param_symbols])

    XspaceErr = np.zeros(len(Xspace))  # Array pre-allocation voor snelheid
    # Gradiëntmatrix met formulae oplossen buiten de loop -> lambda functie
    grad_func = lambdify([x_symbol] + param_symbols, grad, modules="numpy")

    for i, Xs in enumerate(Xspace):
        gradient_np = (
            np.array(grad_func(Xs, *popt)).astype(np.float64).squeeze()
        )
        XspaceErr[i] = math.sqrt(gradient_np @ pcov @ gradient_np.T)
    return XspaceErr


def fit_rating(model_func, X, Y, Yerr, popt):
    """Beoordeel fitkwaliteit op basis van gemiddelde σ-afwijking en χ².

    Parameters
    ----------
        model_func : function
            De gefitte functie: f(x, *popt)
        X : array-like
            Onafhankelijke variabele (x-data)
        Y : array-like
            Afhankelijke variabele (y-data)
        Yerr : array-like
            Meetfouten (1σ onzekerheid per y-punt)
        popt : array-like
            Gefitte parameters

    Returns
    -------
        dict met chi2, chi2_red, avg_sigma
    """
    resid = Y - model_func(X, *popt)
    norm_res = resid / Yerr

    chi2 = np.sum(norm_res**2)
    dof = len(Y) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan
    avg_sigma = np.sqrt(np.mean(norm_res**2))

    # Beoordeling obv σ-afwijking
    if avg_sigma < 0.5:
        print(
            "\n⚠️  Fit te mooi om waar te zijn.",
            "Mogelijk overfitting (teveel parameters) of te coulante fouten",
        )
    elif avg_sigma <= 1.0:
        print("\n⭐  Top fit, komt overeen met meetfouten.")
    elif avg_sigma <= 1.5:
        print(
            "\n✔️  Prima fit, maar valt tussen de onzekerheden.",
            "(Te strakke meetfouten?)",
        )
    elif avg_sigma < 2:
        print(
            "\n❌ Twijfelachtige fit,",
            "slecht model of meetfouten waarschijnlijk te strak.",
        )
    else:
        print(
            "\n❌ Slechte fit o.b.v. meetfouten.",
            "(Een verkeerd/onderfit model of verzonnen meetfouten?)",
        )

    print(f"    Gem. afstand tot fit : {avg_sigma:.2g}σ")
    print(f"    χ²                   : {chi2:.2f}")
    print(f"    χ²_red (DOF={dof})      : {chi2_red:.2f}")

    return {"chi2": chi2, "chi2_red": chi2_red, "avg_sigma": avg_sigma}


def MonteCarlocurvefit(
    model_function, Xdata, Ydata, Xerr, Yerr, p0, runs=1000
):
    """Monte Carlo Curve fitting.

    Normale curvefit maar met gesimuleerde data obv onzekerheid.
    'Wat als ik de meting 1000x zou herhalen waar de fout in de x-as de
    normaalverdeling is van de verschuiving'
    """
    popt_samples = np.zeros((runs, len(p0)))  # Pre-allocation
    valid_count = 0  # Hiermee skipt ie gefaalde fits
    Xdata = np.asarray(Xdata)  # Pre-conversie (optimalisatie)
    Ydata = np.asarray(Ydata)
    Xerr = np.asarray(Xerr)
    Yerr = np.asarray(Yerr)

    print("\nData genereren op basis van de meetfouten:")

    for i in range(runs):
        if (i + 1) % (runs / 20) == 0:  # Statische laadbar per 1/20e iteraties
            filled = int((i + 1) / runs * 20)
            bar = "[" + "|" * filled + "." * (20 - filled) + "]"
            print(f"\r{bar} {i+1}/{runs}", end="", flush=True)

        Xperturbed = Xdata + np.random.normal(0, Xerr)
        Yperturbed = Ydata + np.random.normal(0, Yerr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
        try:
            popti, _ = curve_fit(model_function, Xperturbed, Yperturbed, p0=p0)
            popt_samples[valid_count] = popti
            valid_count += 1  # Update positie voor volgende fit
        except (RuntimeError, TypeError, ValueError):
            continue  # Skip fits die niet convergeren

    popt_samples = popt_samples[:valid_count]  # Snijd de lijst af bij skips

    popt = np.mean(popt_samples, axis=0)
    pcov = np.cov(popt_samples, rowvar=False)
    perr = np.std(popt_samples, axis=0)  # De fout in de parameters

    rating = fit_rating(model_function, Xdata, Ydata, Yerr, popt)
    print("\npopt:", popt)
    print("perr:", perr)

    return popt, pcov, perr, rating


def ODRcurvefit(
    model_function,
    Xdata,
    Ydata,
    Xerr,
    Yerr,
    p0,
    n_starts=10,
    perturb_scale=0.5,
):
    """Handmatige Orthogonal Distance Regression code.

    Ter vervanging van scipy.ODR ivm bizarre instabiliteit en slechte debug
    kwaliteiten. Dit gebruikt scipy.optimize.minimize. Dankjewel Chat (!)

    Dit model weegt X en Y even sterk en is zwaar om te berekenen. Wanneer er
    lineaire afhankelijkheid is tussen parameters, kan het een lokaal minimum
    opleveren van dx^2 +dy^2. De code gebruikt multi-start om dit te verhelpen.

    Gebruik methode 1 of 2 voor gewone/lineaire fits. Hier haal je pas wat uit
    bij nonlineaire functies of kleine datasets waar elk meetpunt zwaar telt.


    n_starts        : int
        Aantal pogingen bij multi-start
    perturb_scale   : float
        Factor van de normaalverdeling std.dev van de ruis voor multi-start.
        Als de fit slecht gaat door vele minima, kan dit naar 1.0+
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            popt_cf, _ = curve_fit(model_function, Xdata, Ydata, p0=p0)
        except (RuntimeError, ValueError, FloatingPointError) as err:
            warnings.warn(
                f"curve_fit faalde: {err}. Gebruik originele p0.",
                RuntimeWarning,
            )
            popt_cf = p0

    print("\nHandmatige p0:", p0)
    print("Curve_fit p0:", [float(f"{v:.4g}") for v in popt_cf])

    def loss_given_p0(p0_used):
        p0_combined = np.concatenate([p0_used, Xdata])

        def orthogonal_loss(p_combined):
            params = p_combined[: len(p0_used)]
            x_perturbed = p_combined[len(p0_used) :]  # noqa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y_model = model_function(x_perturbed, *params)
            # Check output voor inf of nan
            if not np.all(np.isfinite(y_model)):
                return 1e20  # Grote waarde als penalty aka vermijd dit pad

            dx = (x_perturbed - Xdata) / Xerr
            dy = (y_model - Ydata) / Yerr
            return np.sum(dx**2 + dy**2)

        result = minimize(orthogonal_loss, p0_combined, method="L-BFGS-B")
        return result

    # Multi-start logic
    best_result = None
    best_loss = np.inf

    for i in range(n_starts):
        # Laadbalk elke 1/20e van de voortgang
        if (i + 1) % max(1, n_starts // 20) == 0 or i == n_starts - 1:
            filled = int((i + 1) / n_starts * 20)
            bar = "[" + "|" * filled + "." * (20 - filled) + "]"
            print(f"\r{bar} {i + 1}/{n_starts} ", end="", flush=True)

        rel_perturb = np.clip(
            np.abs(popt_cf),
            1e-8,
            None,
        )  # voorkom nul-schaal
        p0_try = popt_cf + np.random.normal(0, rel_perturb * perturb_scale)
        result = loss_given_p0(p0_try)
        if result.success and result.fun < best_loss:
            best_result = result
            best_loss = result.fun
        if not result.success:
            print(
                f"⚠️ {result.message.lower().capitalize()}",
                flush=True,
            )

    if best_result is None:
        print(
            "Geen geldige fit gevonden met multi-start.",
            "\nHier de initiële curve_fit:",
        )
        best_result = loss_given_p0(popt_cf)

    # Ontleed resultaat
    p_opt_combined = best_result.x
    popt = p_opt_combined[: len(popt_cf)]
    x_opt = p_opt_combined[len(popt_cf) :]  # noqa

    # Jacobiaan en foutschatting
    eps = np.sqrt(np.finfo(float).eps)
    J = approx_fprime(popt, lambda p: model_function(x_opt, *p), epsilon=eps)
    residuals = Ydata - model_function(x_opt, *popt)
    dof = max(0, len(Ydata) - len(popt))
    s_sq = np.sum(residuals**2) / dof if dof > 0 else 1.0
    pcov = np.linalg.pinv(J.T @ J) * s_sq
    perr = np.sqrt(np.diag(pcov))

    rating = fit_rating(model_function, Xdata, Ydata, Yerr, popt)
    print("\npopt:", popt)
    print("perr:", perr)

    return popt, pcov, perr, rating


def LeastSquarescurvefit(model_function, Xdata, Ydata, Xerr, Yerr, p0):
    """Ordinary SciPy Curvefitting."""
    # ########### Xerr is omitted  ###########
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("once", RuntimeWarning)
        popt, pcov = curve_fit(
            model_function,
            Xdata,
            Ydata,
            p0=p0,
            sigma=Yerr,
            absolute_sigma=True,
        )
    perr = np.sqrt(np.diag(pcov))

    rating = fit_rating(model_function, Xdata, Ydata, Yerr, popt)
    print("\npopt:", popt)
    print("perr:", perr)
    return popt, pcov, perr, rating


# %% Plotting
"""
#################
#######################       Plotting       #################################
#########
"""


def sigmaPolynoomfit(  # noqa
    ax,
    type_fit,
    Xdata,
    Ydata,
    Xerr=None,
    Yerr=None,
    label="meting 1",
    colour="blue",
    func="a0 * x ** a1 + a2",
    p0=[1, 1, 1],
    regpoints=500,
    sigma=True,
    scatter=True,
    runs=1000,
    sigma_val=3,
    full_label=True,
):
    """Vul de data in een krijg een voorgekauwde fit en grafiek.

    Het is mogelijk de fitfunctie aan te passen naar wens. Let op notatie.
    Kies fitmethode 1, 2 of 3 (zie hieronder).
    Vergeet erna niet om grafiek_opmaak() te gebruiken.
    De regressie gaat van 0 tot 1.2 * max(Xdata), dit is te vinden
    bovenin deze functie.

    Uitleg van de 3 fitmethoden:
        1) Algemene Least Squares fitmethode.

        Werk goed als supersnelle datafit. Deze methode gaat uit van
        verwaarloosbare onzekerheid in de X-as. Dit is veelal een prima
        aanname om mee te werken.

        2) Monte Carlo fitmethode.

        Dit genereert statistische nepdata/ruis in de X én Y-as op basis van de
        puntonzekerheden en doet er een curvefit op. Van alle N fits wordt een
        gemiddelde genomen voor het bepalen van de parameters. Een robuust
        model; faalt zelden en is vrij snel. Met oscillatorische data kan het
        falen. Debuggen van deze methode is goed te doen.

        3) Orthogonal Distance Regression

        Dit is een nauwkeurigere methode. Het neemt onzekerheid van beide
        assen in weging en maakt een curve met maxima en minima. Wanneer het
        convergeert bij een minimum is de fit gevonden. De betrouwbaarheid valt
        wel eens tegen, dus worden er gaussisch N punten omheen gekozen om
        lokale minima te vermijden voor het globale minimum. De methode
        werkt speficiek erg goed bij non-lineaire modellen en kleine datasets
        waar elke meetfout sterk meeweegt. Qua debuggen is het een black box.

    Parameters
    ----------
        ax    : var
            axessubplot (ax1,ax2,...etc.).
        type_fit : int
            1) Nonlinear Least Squares fit
            2) NLS Monte Carlo fit
            3) Orthogonal Distance Regression fit
        Xdata : array
            Datapunten
        Ydata : array
            Datapunten
        Xerr  : array (optioneel indien sigma=False)
            Errorwaarden
        Yerr  : array (optioneel indien sigma=False)
            Errorwaarden
        label : str (opt)
            Metingnaam
        kleur : str (opt)
            Python kleurencode
        func  : str (opt)
            De functie van de regressie in string vorm, werkt met numpy en
            math.\n
            Bijv.: "a0 * x ** a1 + a2"
        p0    : array (niet echt opt)
            De startparameters
        sigma : bool
            Onzekerheidsinterval (intensieve code voor CPU)
        scatter   : bool
            Errorbar plot
        runs   : int (opt)
            Aantal gegenereerde iteraties bij de Monte Carlo fit
            Of het aantal multi-starts bij ODR (runs // 100)
        regpoints : int (opt)
            Het aantal punten voor de regressielijn, fijn om mee te spelen
            bij een zoom-in
        sigma_val : int (1-3)
            Breedte van het betrouwbaarheidsinterval\n
            (68.27% - 95.45% - 99.73%)
        sigma_label : bool
            Keuze tot het verwijderen van het label '3 sigma interval'

    Returns
    -------
        variabelen :
            popt (de optimale parameters)\n
            pcov (covariantie vd parameters)\n
            perr (error/fout id parameters)\n
        functies :
            Plot instellingen (plt.plot() moet nog hierna)\n
            grafiek_opmaak() wordt aangeraden\n
            Dit biedt de mogelijkheid meerdere grafieken in
            dezelfde plot te krijgen
    """

    # Check if amount of parameters matches p0, and if namings are correct
    if not check_for_parameters(p0, func):
        sys.exit()

    if sigma:
        assert len(Ydata) == len(Xdata) == len(Yerr) == len(Xerr)
    else:
        assert len(Ydata) == len(Xdata)

    model_function = set_function_shape(func)  # zet om in lambda functie
    Xspace = np.linspace(  # Sets the range
        min(Xdata) * 1.2 if min(Xdata) < 0 else 0,
        max(Xdata) * 1.2,
        regpoints,
    )

    #########################
    # Choose your pokémon

    if type_fit == 1:
        popt, pcov, perr, rating = LeastSquarescurvefit(
            model_function,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
        )

    elif type_fit == 2:
        popt, pcov, perr, rating = MonteCarlocurvefit(
            model_function,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
            runs,
        )

    elif type_fit == 3:
        popt, pcov, perr, rating = ODRcurvefit(
            model_function,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
            runs // 100,
        )

    #########################

    if sigma:
        # Het onzekerheidsgebied dmv SymPy berekenen
        # Partiëel differentiëren per parameter en covariantie matrix opstellen

        print("Onzekerheidsinterval berekenen...", end="")
        # Dit is de sauce
        XspaceErr = uncertainty_range(func, popt, pcov, Xspace)
        # Sigma gebied
        ax.fill_between(
            Xspace,
            model_function(Xspace, popt) + sigma_val * XspaceErr,
            model_function(Xspace, popt) - sigma_val * XspaceErr,
            facecolor=colour,
            alpha=0.2,
            label=(
                r"$%d\sigma$ interval" % (sigma_val) if full_label else None
            ),
        )
        foutmarge = abs(
            (sigma_val * XspaceErr[0]) * 100 / model_function(Xspace, popt)[0],
        )
        print(f"\r{sigma_val}σ_%(x=0) = {foutmarge:.2g}%" + " " * 20)

    if scatter:
        # Meetpunten met hun foutmarges
        ax.errorbar(
            Xdata,
            Ydata,
            xerr=Xerr,
            yerr=Yerr,
            linewidth=0.5,
            capsize=3,
            fmt=".",
            color=colour,
            label=label.capitalize(),
        )

    # De plot van de regressie met een verwerkt en opgemaakt label
    regressionlabel = Reglabelmaker(func, popt, perr, full_label)
    ax.plot(
        Xspace,
        model_function(Xspace, popt),
        linewidth=0.5,
        linestyle="-",
        color=colour,
        label=regressionlabel % label if full_label else regressionlabel,
    )

    return popt, perr, rating  # , pcov


def grafiek_opmaak(
    ax,
    xlabel="x",
    ylabel="y",
    legendlocation=4,
    xlim=None,
    ylim=None,
    grid=True,
    ncol=2,
):
    r"""Doet wat het zegt. Standaard opmaakset voor grafieken.

    Is NIET inclusief plt.plot()

    Parameters
    ----------
        ax : var
            axessubplot (ax1,ax2,...etc.).
        xlabel : r-str
            Voorbeeld: r"$\frac{1}{4\pi^2m}  \[\frac{1}{kg}\]$".
            Of: r"$V$ [m$^3$]"
        ylabel : r-str
        legendlocation : int, str
            Default is rechtsonder, google matplotlib.pyplot.legend(loc='x').
            5 is eigen additie voor de legenda onder de grafiek.
        xlim : list

        ylim : list

        grid : bool

        ncol : int
            Aantal kolommen bij legendlocation = 5

    Returns
    -------
        None
    """
    ax.set_xlabel(xlabel, labelpad=5, fontsize=14)
    ax.set_ylabel(ylabel, labelpad=5, rotation=90, fontsize=14)
    if legendlocation != 5:
        ax.legend(loc=legendlocation)
    else:
        # Shrink current axis' height by 10% on the bottom
        box = ax.get_position()
        ax.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9],
        )

        # Put a legend below current axis
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=ncol,
        )
    if grid:
        ax.grid()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


# %% Main

if __name__ == "__main__":
    import types

    def is_user_func(name):
        """Docstring."""
        obj = globals().get(name)
        return (
            isinstance(obj, types.FunctionType)
            and obj.__module__ == "__main__"
            and hasattr(obj, "__code__")
            and name != "is_user_func"  # exclude itself
        )

    user_funcs = [name for name in dir() if is_user_func(name)]

    # Sort functions by definition line number
    user_funcs.sort(key=lambda name: globals()[name].__code__.co_firstlineno)

    print("Functies in dit bestand:")
    for f in user_funcs:
        print("- " + f)
    print("\nTyp help(tn.<functie>) voor documentatie")
