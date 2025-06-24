"""
Kleine module met functies die mij teveel tijd hebben gekost.
Om het te gebruiken in andere bestanden zie de voorbeeldbestanden die
meegegeven zijn.
Floris Messack, HHS : 2023
"""

import math
import re as re2  # naming overlap (extracting_variables functie)
import string  # extracting_variables
import sys  # Voor de abort functie bij een error
from pathlib import Path  # voor vinden van downloads folder

import numpy as np
import pandas as pd  # Voor het inlezen van de excel
from scipy.odr import Model
from scipy.odr import ODR
from scipy.odr import RealData
from scipy.optimize import curve_fit  # Voor de standaard curvefit
from sympy import diff  # Voor het onzekerheidsgebied van de functie
from sympy import Matrix
from sympy import re
from sympy import symbols
from sympy import sympify


ManualError = 1  # error handling
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


def lees_bestand(file, sheet="Sheet1", cols=["U", "I"], debug=0):
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
    debug : int (opt)
        1 of 0 als je de volledige pandas dataframe wilt van de excel

    Returns
    -------
    dict : (def)
        Opgevraagde kolommen als één numpy dictionary
    dict : (opt)
        De volledige pandas dataframe van de excel en de NaN-count per kolom
    """
    file = Path(file)
    file_downl = Path(f"{Path.home()}/Downloads/{file}")

    for thefile in [file, file_downl]:  # Bestandsfolder heeft voorrang
        if thefile.is_file():
            Excel = pd.read_excel(thefile, sheet)
            print(f"\nBestand {thefile} gevonden!")
            break
    else:
        print(
            f"\nBestand niet gevonden in dezelfde folder of in "  # noqa
            + f"{Path.home()}\\Downloads !\nAborted script",  # noqa
        )  # noqa
        sys.exit()  # abort code

    print(f"Zoekend naar kolom(men) {cols} in sheet: '{sheet}'...")

    if debug == 1:
        return zoek_kolom(Excel, cols, 1), Excel
    else:
        return zoek_kolom(Excel, cols)


def zoek_kolom(diction, col, debug=0):
    """
    Kolommen vinden in een panda's df en vertalen naar Numpy, met errorfunctie.

    Parameters
    ----------
    dictionary : dict
        De dictionary waarin gezocht moet worden
    col : str
        kolomtitel(s) die gezocht moet(en) worden
    debug : (int)
        1 of 0 als je de wilt weten hoeveel NaNs

    Returns
    -------
    newdict : dict
        met specifiek die kolommen
    print (opt)
        NaN count
    """
    newdict = {}
    for column in col:
        try:
            # Zoek de kolom met metingen en creëer een dict entry met deze data
            newdict.update({column: diction[column].dropna().to_numpy()})
            if (diction[column].isnull().values.sum() != 0) and (debug == 1):
                print(
                    "-debug- Kolom '{}' bevat {} NaN(s)".format(
                        column,
                        diction[column].isnull().values.sum(),
                    ),
                )
        except KeyError:
            print(f"Kolom '{column}' niet gevonden!\n")
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
        except OverflowError as er:
            print(
                "\nERROR:",
                er,
                "\nEr is een 'inf' gegeven als "
                + ("meetwaarde" if n == 0 else "onzekerheid")
                + " voor significantie afronding. "
                + "Check de input voor deze functie. \n",
            )
            raise ManualError
    elif number == 0:  # Bij een nul moet ie niet kutten
        return "0"
    else:
        return "%s" % number  # Bij een onzekerheid van nul is het gegeven

    power = numpow - sigpow  # the difference of orders
    if power < 0:
        print(
            f"\nSignificantie error: De onzekerheid {sig_number:.3e} is te "
            + f"groot voor de gegeven meetwaarde van {number:.3e} !\n",
        )

        raise ManualError
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
    # except AttributeError:
    #     print(
    #         "ERROR: De fitfunctie is incorrect ingevoerd, gebruik spaties!\n"
    #     )
    #     sys.exit()
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
            raise ValueError(
                "Parameter values must be provided",
                "if replace_with_values=True.",
            )

        # Check if number of values matches number of variables (excluding 'x')
        if len(popt) != len(mapping) - 1:
            raise ValueError(
                "The number of values given does",
                "not match the number of parameters.",
            )

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


def set_function_shape(formula, ODR=False):
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
    vars.insert(0, "x")

    safe_globals = {"np": np, "math": math}

    if ODR:
        return lambda beta, x: eval(
            formula,
            safe_globals,
            {**dict(zip(vars[1:], beta)), "x": x},
        )

    def func(x, *beta):
        if isinstance(
            beta[0],
            np.ndarray,
        ):  # If beta[0] is an array inside a tuple
            beta = beta[0]  # Extract the array directly

        local_dict = dict(zip(vars[1:], beta))
        local_dict["x"] = x
        return eval(formula, safe_globals, local_dict)

    return func


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
    Plots a 3 sigma confidence band using Monte Carlo simulation.

    Parameters:
        ax      : figure nameselection
        func    : Callable. Must accept (p1, p2, ..., x) and return an array.
        popt    : List or array of best-fit parameters [p1, p2, ...].
        perr    : List or array of 1-sigma uncertainties [dp1, dp2, ...].
        x_vals  : Array of x values to evaluate the function over.
        colour  : String. Chooses the colour of the plot.
        N       : Number of Monte Carlo samples (default 1000).
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
            print(
                "Value %s=%d is out of range for the MMTTi, " % (UI, data)
                + "data is incorrect.",
            )
            sys.exit()

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
        raise TypeError(
            "MMTTi didnt receive workable instance, use int, dict or list."
            " Received %s" % type(meetdata),
        )


# %% Dataverwerking subfuncties
"""
#################
#######################      Subfuncties     #################################
#########
"""


def uncertainty_range(formula, popt, pcov, Xspace):
    """3sigma interval met SymPy, accepteert een variabel aantal parameters(!).

    Stappenplan:
        Haalt de variabelen uit de formule en hernoemt ze a-z
        definiëert x als variabele, rest als parameters
        Fixt wat standaard syntax-moeilijkheden.
        Doet partiële differentiatie van de formule t.o.v. elke parameter.
        Creëert een gradiëntmatrix van alle parameters. Daarna ben ik het
        vergeten, iets met de covariantiematrix en dan over de hele linspace.
        Geeft het onzekerheidsgebied terug.

    Parameters
    ----------
        formula : string
                    "a0/a1 * (1 - np.exp(-x/a1) + a2" doe eens gek
        popt : list
            Optimale parameters van de voorafgaande fit
        pcov : 2D-list
    Returns
    -------
        XspaceErr : list
            De grootte van de afwijking van de fit over de linspace
    """
    param_string = extract_variables(formula)  # "a b x" used for symbols()
    param_names = param_string.split()  # ['a', 'b', 'x'] used for making dict

    # Convert all variable names to sympy symbols
    symbol_list = symbols(param_string)  # (a, b, x)   # Now defined as symbols
    symbol_dict = dict(zip(param_names, symbol_list))  # For quick expr cleanup

    x_symbol = symbol_dict["x"]

    # Speciale functies zoals sin, exp, log10 naar sympy krijgen
    clean_formula = (
        re2.sub(r"log(\d+(?:\.\d+)?)\(", r"(1/log(\1))*log(", formula)
        .replace("math.", "")
        .replace("np.", "")
    )
    # Sympify the expression
    sympy_expr = sympify(clean_formula, locals=symbol_dict)
    # Compute gradient (partial derivatives w.r.t. each parameter except x)
    param_symbols = [symbol_dict[v] for v in param_names if v != "x"]
    grad = Matrix([diff(sympy_expr, p) for p in param_symbols])

    XspaceErr = np.zeros(len(Xspace))  # Array pre-allocation

    for i, Xs in enumerate(Xspace):
        subs_dict = {x_symbol: Xs}
        subs_dict.update({p: val for p, val in zip(param_symbols, popt)})
        gradient = grad.subs(subs_dict).evalf()
        gradient_np = np.array(gradient).astype(np.float64).squeeze()
        XspaceErr[i] = math.sqrt(gradient_np @ pcov @ gradient_np.T)
    return XspaceErr


def MonteCarlocurvefit(formula, Xdata, Ydata, Xerr, Yerr, p0, runs=1000):
    """Monte Carlo Curve fitting.

    Normale curvefit maar met gesimuleerde data obv onzekerheid.
    """
    model_function = set_function_shape(formula)  # Creates a workable function
    popt_samples = []  # Introducing list

    print("\nData genereren op basis van de meetfouten:")
    for i in range(runs):
        # Perturb both x and y based on their respective uncertainties
        Xperturbed = Xdata + np.random.normal(0, Xerr)
        Yperturbed = Ydata + np.random.normal(0, Yerr)

        try:
            print(f"\r{i+1}/{runs}  ", end="")  # Static loading bar
            popti, _ = curve_fit(model_function, Xperturbed, Yperturbed, p0=p0)
            if len(popti) == len(p0):  # Keeps out addressing errors
                popt_samples.append(popti)
        except (RuntimeError, TypeError):
            # Skip fits die niet convergeren
            continue

    popt_samples = np.array(popt_samples)

    popt = np.mean(popt_samples, axis=0)
    pcov = np.cov(popt_samples, rowvar=False)
    perr = np.std(popt_samples, axis=0)  # De fout in de parameters

    print("\n\npopt:", popt)
    print("perr:", perr)
    # print("pcov:\n", pcov)

    #########################
    return model_function, popt, pcov, perr


def ODRcurvefit(formula, Xdata, Ydata, Xerr, Yerr, p0):
    """Orthogonal Distance Regression Curve fitting."""
    # Improve the p0 guess to better chances of good fit.
    model_function = set_function_shape(formula, ODR=False)
    popt, _ = curve_fit(model_function, Xdata, Ydata, p0=p0)

    print("Initial p0:", p0)
    print("Scipy.curve_fit p0:", popt)

    # Preparing ODR fit
    model_function = set_function_shape(formula, ODR=True)
    data = RealData(Xdata, Ydata, Xerr, Yerr)  # Creating the dataset
    model = Model(model_function)

    odr = ODR(data, model, beta0=popt)  # Defining the fit

    odr.set_job(fit_type=2)  # Setting to real ODR
    output = odr.run()

    pcov = output.cov_beta  # Covariance matrix
    perr = np.sqrt(np.diag(pcov))  # Error in the parameters
    popt = output.beta  # Optimal parameters (fit)

    print("Converged?", output.info, "tries")  # nr of tries
    # If pcov is [[0,0],[0,0]], then it's probably correlated
    # print("Covariance:", pcov)
    print("Beta:", popt)

    return model_function, popt, pcov, perr


def LeastSquarescurvefit(formula, Xdata, Ydata, Xerr, Yerr, p0):
    """Ordinary SciPy Curvefitting."""
    # ########### Xerr is omitted  ###########

    model_function = set_function_shape(formula, ODR=False)

    popt, pcov = curve_fit(
        model_function,
        Xdata,
        Ydata,
        p0=p0,
        sigma=Yerr,
        absolute_sigma=True,
    )

    # Standard Deviation
    perr = np.sqrt(np.diag(pcov))
    print("\npopt:", popt)
    print("perr:", perr)
    return model_function, popt, pcov, perr


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
    MC_runs=1000,
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

        Deze methode gaat uit van GEEN onzekerheid in de x-as. Dit is veelal
        een prima aanname om mee te werken, zie het voorbeeld bij methode 3

        2) Monte Carlo fitmethode.

        Dit gebruikt een normaalverdeling van de onzekerheid om nepdata
        te genereren. Op basis hiervan wordt een fit gegenereerd met een
        gaussische onzekerheid in beide parameters.
        Werkt ook wanneer x en y direct gecorreleerd zijn

        3) Orthogonal Distance Regression

        Dit is een nauwkeurigere methode. Het neemt onzekerheid van beide
        assen in weging, maar valt uiteen wanneer in de meting de assen
        gecorreleerd zijn. Denk aan een snelheidsmeting, waarbij
        meetonzekerheid van de positie én de tijd afhankelijk is. Dit valt
        samen te voegen zodat er maar één de onzekerheid heeft.
        Gebruik dan methode 1 of 2.
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
        MC_runs   : int (opt)
            Aantal gegenereerde iteraties bij de Monte Carlo fit
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

    Xspace = np.linspace(  # Sets the range
        min(Xdata) * 1.2 if min(Xdata) < 0 else 0,
        max(Xdata) * 1.2,
        regpoints,
    )

    #########################
    # Choose your pokémon

    if type_fit == 1:
        model_function, popt, pcov, perr = LeastSquarescurvefit(
            func,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
        )

    elif type_fit == 2:
        model_function, popt, pcov, perr = MonteCarlocurvefit(
            func,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
            MC_runs,
        )

    elif type_fit == 3:
        model_function, popt, pcov, perr = ODRcurvefit(
            func,
            Xdata,
            Ydata,
            Xerr,
            Yerr,
            p0,
        )

    #########################

    if sigma:
        # Het onzekerheidsgebied dmv SymPy berekenen
        # Partiëel differentiëren per parameter en covariantie matrix opstellen

        print("\nOnzekerheidsinterval berekenen...", end="")
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
        print(f"\r{sigma_val} sigma foutmarge op y(0)={foutmarge:.1e}%")

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

    return popt, pcov, perr


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
