import numpy as np

"""
Onnauwkeurigheid S120C:
https://www.thorlabs.com/drawings/d6b87b6c3bfc3386-781D95AB-E08D-414E-01A2638FB3323F0E/S120C-SpecSheet.pdf
Responsiviteitsdata S120C
https://www.thorlabs.com/images/tabimages/S120C_Responsivity.xlsx
Onnauwkeurigheidstabel PM100D
https://plexon.com/wp-content/uploads/2017/06/ThORLABS-PM100D-Power-Meter-Operation-Manual.pdf

Code kan nog uitgebreid worden met een nested PM100D dict voor de Thermal
sensor en Pyroelectrical sensor
"""


def S120C_error(meetdata, golflengte_nm):
    """Functie om de fout in de meting te bepalen van de ThorLabs S120C.

    Werkt per waarde of verwerkt dict met 1-2 kolommen

    Geeft meetdata terug met 1 significant getal, zoals onzekerheden horen.

    Parameters
    ----------
        meetdata : int, float, labeled array OR dict

    Returns
    -------
        data_verwerkt : list OR dict
            Onzekerheid van de meting
    """
    S120C_error_array = {
        "wavelength": [440, 981, 1101],  # nm   Range = 400-1100 inclusief
        "read_accuracy": [0.05, 0.03, 0.07],  # %
    }

    PM100D_error_array = {
        "range": [50e-9, 500e-9, 5e-6, 50e-6, 500e-6, 5e-3],  # A
        "f.s. accuracy": [0.005, 0.002, 0.002, 0.002, 0.002],  # %
        "resolutie": [10e-12, 100e-12, 1e-9, 10e-9, 100e-9, 1e-6],  # A
    }

    S120C_responsivity = [  # Wavelength (nm), Responsivity (mA/W)
        [400, 22.22292],
        [405, 22.77137],
        [410, 23.29455],
        [415, 23.77271],
        [420, 24.20838],
        [425, 24.61039],
        [430, 24.99048],
        [435, 25.35691],
        [440, 25.70098],
        [445, 26.01537],
        [450, 26.31522],
        [455, 26.61524],
        [460, 26.90603],
        [465, 27.17678],
        [470, 27.43527],
        [475, 27.69095],
        [480, 27.94135],
        [485, 28.18324],
        [490, 28.42224],
        [495, 28.66340],
        [500, 28.90072],
        [505, 29.12758],
        [510, 29.34610],
        [515, 29.55957],
        [520, 29.76752],
        [525, 29.96927],
        [530, 30.16711],
        [535, 30.36297],
        [540, 30.55415],
        [545, 30.73822],
        [550, 30.91809],
        [555, 31.09698],
        [560, 31.27384],
        [565, 31.44774],
        [570, 31.62251],
        [575, 31.80082],
        [580, 31.97617],
        [585, 32.14184],
        [590, 32.29965],
        [595, 32.45357],
        [600, 32.60792],
        [605, 32.76573],
        [610, 32.92459],
        [615, 33.08151],
        [620, 33.23657],
        [625, 33.39039],
        [630, 33.54264],
        [635, 33.69351],
        [640, 33.84602],
        [645, 34.00262],
        [650, 34.16036],
        [655, 34.31603],
        [660, 34.47063],
        [665, 34.62550],
        [670, 34.77910],
        [675, 34.93031],
        [680, 35.08255],
        [685, 35.24030],
        [690, 35.40760],
        [695, 35.58690],
        [700, 35.77459],
        [705, 35.96663],
        [710, 36.16314],
        [715, 36.36558],
        [720, 36.57643],
        [725, 36.79705],
        [730, 37.02300],
        [735, 37.24945],
        [740, 37.47566],
        [745, 37.70254],
        [750, 37.93346],
        [755, 38.17164],
        [760, 38.41735],
        [765, 38.66922],
        [770, 38.92219],
        [775, 39.17165],
        [780, 39.41830],
        [785, 39.66423],
        [790, 39.91184],
        [795, 40.16254],
        [800, 40.41371],
        [805, 40.66243],
        [810, 40.90885],
        [815, 41.15428],
        [820, 41.40181],
        [825, 41.65474],
        [830, 41.91554],
        [835, 42.18412],
        [840, 42.45099],
        [845, 42.70817],
        [850, 42.96318],
        [855, 43.22347],
        [860, 43.48093],
        [865, 43.72710],
        [870, 43.96790],
        [875, 44.21153],
        [880, 44.46114],
        [885, 44.71467],
        [890, 44.95435],
        [895, 45.16330],
        [900, 45.34370],
        [905, 45.50524],
        [910, 45.66834],
        [915, 45.85312],
        [920, 46.06770],
        [925, 46.31054],
        [930, 46.55357],
        [935, 46.76878],
        [940, 46.95509],
        [945, 47.11588],
        [950, 47.24550],
        [955, 47.33340],
        [960, 47.35830],
        [965, 47.30108],
        [970, 47.16177],
        [975, 46.94110],
        [980, 46.62336],
        [985, 46.18961],
        [990, 45.62436],
        [995, 44.91171],
        [1000, 44.03059],
        [1005, 42.96018],
        [1010, 41.68563],
        [1015, 40.19609],
        [1020, 38.49069],
        [1025, 36.57380],
        [1030, 34.46054],
        [1035, 32.17398],
        [1040, 29.75811],
        [1045, 27.25417],
        [1050, 24.67156],
        [1055, 22.04691],
        [1060, 19.55754],
        [1065, 17.37831],
        [1070, 15.53370],
        [1075, 13.99003],
        [1080, 12.63145],
        [1085, 11.35059],
        [1090, 10.15608],
        [1095, 9.068980],
        [1100, 8.043880],
    ]

    def ThorLabs_S120C_error(power, golflengte_nm):
        """De absolute error, berekend dmv responsiviteit en tabellen.

        Gebruikt de nauwkeurigheidstabellen van beide de S120C en de PM100D
        """

        responsivity = 0  # function wide scope
        for wavelength, responsivity in S120C_responsivity:
            if wavelength == golflengte_nm:
                responsivity *= 1e-3  # Tabel is in mA
                break
        else:
            raise UnboundLocalError(
                "⚠️ {golflengte_nm} nm valt buiten het bereik van de S120C"
                + "sensor (400-1100 nm)."
            )
        for index, range_max_wav in enumerate(S120C_error_array["wavelength"]):
            if golflengte_nm < range_max_wav:  # Kleiner = in range
                reading_accuracy_value = S120C_error_array["read_accuracy"][
                    index
                ]

                # relatieve fout toegepast op power
                S120C_power_error_abs = power * reading_accuracy_value

                # Heen en weer omrekenen om PM100D_error te berekenen
                amps = power * responsivity
                PM100D_power_error_abs = PM100D_error(amps) / responsivity

                # Fouten bij elkaar optellen
                power_full_error_abs = np.sqrt(
                    S120C_power_error_abs**2 + PM100D_power_error_abs**2
                )
                return power_full_error_abs
        else:
            raise UnboundLocalError("Idk of je deze ooit gaat tegenkomen")

    def PM100D_error(sensor_data_amps):
        """Gebruikt de output van de sensor voor de onnauwkeurigheidstabellen.

        Geeft de absolute fout terug.
        Input is in ampere!
        De output is ook in ampere.
        Dit moet worden opgeteld bij de sensorwaarde a=sqrt(b^2+c^2)
        """
        for index, range_max_value in enumerate(PM100D_error_array["range"]):
            if sensor_data_amps < range_max_value:
                # Zodra het kleiner is heeft ie de range
                full_rng_accuracy_val = PM100D_error_array["f.s. accuracy"][
                    index
                ]
                resolutie_value = PM100D_error_array["resolutie"][index]
                break
        try:
            PM100D_error_amps_abs = (
                range_max_value * full_rng_accuracy_val + resolutie_value / 2
            )
        except UnboundLocalError:
            raise UnboundLocalError(
                "⚠️ I=%d A is boven het bereik van de PM100D"
                % (sensor_data_amps)
                + f" (max={PM100D_error_array['range'][-1]*1000} mA)",
            )
        return PM100D_error_amps_abs

    # Handling van verschillende inputs, scheelt koppijn
    if isinstance(meetdata, (int, float)):
        # Standaard werking van functie
        return ThorLabs_S120C_error(meetdata, golflengte_nm)

    elif isinstance(meetdata, (np.ndarray, list)):
        # Ingebouwde loop voor arrayverwerking
        meetdata_err = np.zeros(len(meetdata))
        for i, meting in enumerate(meetdata):
            meetdata_err[i] = ThorLabs_S120C_error(meting, golflengte_nm)
        return meetdata_err

    elif isinstance(meetdata, dict):
        print("S120C_error() ontving dictionary")
        # Ingebouwde loop voor dictionaryverwerking, gebruik dict.update()
        meetdata_err = {}
        for kolom in meetdata:
            print("Found", kolom)
            meetdata_err.update(
                {"err_" + kolom: np.zeros(len(meetdata[kolom]))},
            )
            for i, meting in enumerate(meetdata[kolom]):
                meetdata_err["err_" + kolom][i] = ThorLabs_S120C_error(
                    meting, golflengte_nm
                )
        return meetdata_err
    else:
        raise TypeError(
            "S120C_error() verwacht een int/float, dict of list.",
            "Ontving %s" % type(meetdata),
        )
