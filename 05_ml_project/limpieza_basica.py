from sklearn.base import BaseEstimator, TransformerMixin

class LimpiezaBasica(BaseEstimator, TransformerMixin):
    """
    Clase transformadora para limpieza básica de datos de predicción de impagos.
    Compatible con sklearn Pipeline.
    """

    def __init__(self):
        # Columnas que queremos conservar tras la limpieza
        self.columns_to_keep = [
            'UsoCrédito', 'Edad', '30-59DíasTarde', '60-89DíasTarde',
            '90DíasTarde', 'RatioDeuda', 'IngresoMensual',
            'LíneasCrédito', 'PréstamosCasa', 'Dependientes'
        ]

    def fit(self, X, y=None):
        """
        Aprende los valores de imputación (media y moda).
        """
        self.ingreso_mean_ = X['MonthlyIncome'].mean()
        self.dep_mode_ = X['NumberOfDependents'].mode()[0]
        return self

    def transform(self, X):
        """
        Aplica renombrado, imputación, filtrado de edad, y eliminación de duplicados.
        """
        df = X.copy()

        df = df.rename(columns={
            'SeriousDlqin2yrs': 'Impago2Años',
            'RevolvingUtilizationOfUnsecuredLines': 'UsoCrédito',
            'age': 'Edad',
            'NumberOfTime30-59DaysPastDueNotWorse': '30-59DíasTarde',
            'NumberOfTime60-89DaysPastDueNotWorse': '60-89DíasTarde',
            'NumberOfTimes90DaysLate': '90DíasTarde',
            'DebtRatio': 'RatioDeuda',
            'MonthlyIncome': 'IngresoMensual',
            'NumberOfOpenCreditLinesAndLoans': 'LíneasCrédito',
            'NumberRealEstateLoansOrLines': 'PréstamosCasa',
            'NumberOfDependents': 'Dependientes'
        })

        df = df.drop(columns=["Unnamed: 0"], errors='ignore')
        df = df[df['Edad'] >= 18]
        df['IngresoMensual'] = df['IngresoMensual'].fillna(self.ingreso_mean_)
        df['Dependientes'] = df['Dependientes'].fillna(self.dep_mode_)
        df = df.drop_duplicates()

        # ⛑️ Comprobamos si es entrenamiento (tiene la columna objetivo)
        if 'Impago2Años' in df.columns:
            return df[['Impago2Años'] + self.columns_to_keep]
        else:
            return df[self.columns_to_keep]
