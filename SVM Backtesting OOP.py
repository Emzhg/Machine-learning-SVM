from calendar import calendar
from collections import deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from typing import List, Union, Set, Generic, TypeVar
from functools import wraps
import pandas as pd
import time
import requests
import yfinance as yf

"""
-Enum : implémentation des API nécessaires à l'extraction de données.
L'extraction des données sera réalisée sur yahoo finance et cryptocompare

-datetime: module fournissant des classes pour manipuler les dates et heures. Via from datetime import datetime,
on importe directement l'objet datetime. Cela permet de ne pas repréciser sa provenance. 

-La bibliothèque Pandas produit des bases de données structurées faciles à utiliser ainsi que des
outils d’analyse

-time : Ce module fournit diverses fonctions liées au temps notamment pour convertir les dates issues de Yahoo

-yfinance : yahoo finance API python
"""


class WeightId:
    def __init__(self,
                 product_code: List[str] = None,
                 underlying_code: str = None,
                 dt: datetime = None,
                 ):
        self.id = id
        self.product_code = product_code
        self.underlying_code = underlying_code
        self.dt = dt

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash((self.product_code, self.dt, self.underlying_code))


class Weight:
    def __init__(self,
                 id: WeightId,
                 value=float,
                 ):
        self.id = id
        self.value = value

    def __repr__(self):
        return self.__dict__

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash(self.id)


class Params:

    def __init__(self,
                 start_ts: datetime,
                 end_ts: datetime,
                 underlying_tickers: List[str],
                 strategy_name: str,
                 weights: List[Weight]):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.underlying_tickers = underlying_tickers
        self.strategy_name = strategy_name
        self.weights = weights

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)

class params:
    def __init__(self, strat: str = None, mesure_de_risque: str = None, start_dt: datetime = None,
                 end_dt: datetime = None):
        self.strat = strat
        self.mesure_de_risque = mesure_de_risque
        self.start_dt = start_dt
        self.end_dt = end_dt

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)



class Provider(Enum):
    YAHOO = "YAHOO"
    CRYPTOCOMPARE = "CRYPTOCOMPARE"

    """
    liste des provider de données
    """


class Data:

    def __repr__(self):
        kwargs = [f"{key}={value!r}" for key, value in self.__dict__.items() if key[0] != "_" or key[:2] != "__"]
        return "{}({})".format(type(self).__name__, "".join(kwargs))

    """
    -rejet des clefs avec underscore simple ou double lors du prélevement 
    de la liste d'arguments de longueur variables contenant les mots clefs
    -L'attribut __name__ donne le nom originellement attribué à la classe
    join permet de convertir le dictionnaire de kwargs en string
    """


class QuoteId(Data):
    def __init__(self,
                 product_code: List[str] = None,
                 dt: datetime = None,
                 provider: Provider = None
                 ):
        if isinstance(provider, Provider):
            self.provider = provider
        else:
            raise TypeError(f"self.provider must be an instance of Provider")
        self.product_code = product_code
        self.dt = dt

    """
    -création du constructeur de la classe fille QuoteId appelant la classe mère data contenant les arguments.
    -product code permet de choisir le nom du produit financier désiré tel que BTC_USD
    -datetime permet de traiter les dates 
    
    -on appelle la classe provider dans quote id
    -La fonction isinstance() retourne True si l’objet spécifié est du type spécifié, sinon False
    """

    def __repr__(self):
        return str(vars(self))

    """
    -La méthode __repr__ printe l’objet donné.
    -La fonction vars() retourne l’attribut __dict__ de l’objet donné et est donc identique à self. __dict__
    """

    def __str__(self):
        return repr(self)

    """
    -La méthode __str__ représente repr comme un string
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

    """
    - La méthode __eq__ permet de comparer deux variables d'un meme type en vérifiant un paramètre
    """

    def __hash__(self):
        return hash((self.product_code, self.dt, self.provider))

    """
   La méthode hash permet d'utiliser des nombres entiers 
    pour comparer les clés de dictionnaire 
    """


class Quote(Data):
    def __init__(self, id: QuoteId, open: float = None, high: float = None,
                 low: float = None, close: float = None,
                 adjclose: float = None, volume: int = None):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume
        self.id = id

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, Quote) else False

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, **kwargs):
        product_code = params.underlying_tickers
        open = kwargs.get("Open")
        high = kwargs.get("High")
        low = kwargs.get("Low")
        close = kwargs.get("Close")
        adjclose = kwargs.get("Adj Close")
        volume = kwargs.get("Volume")
        dt = pd.to_datetime(end)-pd.to_datetime(begin)
        id = QuoteId(product_code, dt, Provider.YAHOO)
        return cls(open=open, high=high, low=low, close=close, adjclose=adjclose, volume=volume, id=id)

    """
    le décorateur permet de convertir from_json en méthode de classe. Grâce à cela, la méthode from_json 
    prend comme premier paramètre cls qui correspond à la classe Quote et retourne le constructeur
    
    -remplacement de datetime.datetime object par pandas timestamp pour exploiter le json
    -l'id donne les caractéristiques du produit dans le json
    -close est la valeur associée à la clef Close dans les kwargs
    -get retourne la valeur de la clé si la clé est dans le dictionnaire
    
    """

    @staticmethod
    def get_data(json_format: List[dict]):
        return list(map(lambda obj: Quote.from_json(**obj), json_format))

    """
    -retourne un json sous forme de liste à partir des informations de la classe Quote
    """

class QuoteView:
    def __init__(self, product_code: List[str] = None, dt: datetime = None, provider: Provider = None,
                 open: float = None, high: float = None, low: float = None, close: float = None, adjclose: float = None,
                 volume: int = None):
        self.product_code = product_code
        self.dt = dt
        self.provider = provider
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume

    """
    -classe permettant d'identifier les composantes à conserver dans le json
    """

    @classmethod
    def from_quote(cls, quote: Quote):
        return cls(product_code=quote.id.product_code, dt=quote.id.dt, provider=quote.id.provider, open=quote.open,
                   high=quote.high, low=quote.low, close=quote.close, adjclose=quote.adjclose, volume=quote.volume)

    """
    -méthode de classe permettant de dérouler les attributs d'une liste définis par QuoteView
    """


class Factory:

    @staticmethod
    def to_quote_view(quotes: list([Quote])):
        return list(map(lambda quote: QuoteView.from_quote(quote), quotes))

    """
    -transformation de la liste en décomposant id en 3 attributs
    """


class Data_json:

    def __init__(self, df_select):
        self.df_select = df_select


    @staticmethod
    def convert_data(df_select):
        df_T = df_select.T
        dict_opt = df_T.groupby(level=0).apply(lambda df: df.xs(df.name).to_dict(orient="index")).to_dict()
        json_format = [value for value in dict_opt.values()]
        quote_list = Quote.get_data(json_format)
        quote_1D = Factory.to_quote_view(quote_list)
        return quote_1D

    """
    -conversion de la dataframe en dictionnaire
    -analyse clef par clef du dictionnaire json
    -récupération des quotes dans le json sous forme de liste
    -création d'une liste avec le déroulé des attributs nommées dans la classe QuoteView
    -représentation de la liste en dataframe et tri
    """

class LinearSvmClassifier:
    
    def __init__(self, C):
        self.C = C                                 
        self.alpha = None
        self.w = None
        self.supportVectors = None
    
    def fit(self, X, y):
        N = len(y)
        # Gram matrix of (X.y)
        Xy = X * y[:, np.newaxis]
        GramXy = np.matmul(Xy, Xy.T)

        # Lagrange dual problem
        def Ld0(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))

        # Partial derivate of Ld on alpha
        def Ld0dAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        A = np.vstack((-np.eye(N), np.eye(N)))             # <---
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))  # <---
        constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

        # Maximize by minimizing the opposite
        optRes = optimize.minimize(fun=lambda a: -Ld0(GramXy, a),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda a: -Ld0dAlpha(GramXy, a), 
                                   constraints=constraints)
        self.alpha = optRes.x
        self.w = np.sum((self.alpha[:, np.newaxis] * Xy), axis=0)  
        epsilon = 1e-6
        self.supportVectors = X[self.alpha > epsilon]
        # Support vectors is at a distance <= 1 to the separation plan
        # => use min support vector to compute the intercept, assume label is in {-1, 1}
        signedDist = np.matmul(self.supportVectors, self.w)
        minDistArg = np.argmin(signedDist)
        supportLabels = y[self.alpha > epsilon]
        self.intercept = supportLabels[minDistArg] - signedDist[minDistArg]
    
    def predict(self, X):
        """ Predict y value in {-1, 1} """
        assert(self.w is not None)
        assert(self.w.shape[0] == X.shape[1])
        return 2 * (np.matmul(X, self.w) > 0) - 1

class SVM: #https://www.kaggle.com/migom6/svm-with-smo-from-scratch
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel_type='linear'):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_polynomial,
            'gaussian' : self.kernel_radial
        }
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def kernel_linear(self, x_1, x_2):
        X = LinearSvmClassifier(C=1)

    def kernel_polynomial(self, x_1, x_2):
        pass

    def kernel_radial(self, x_1, x_2,sigma=1): #Radial basis function kernel (RBF)/ Gaussian Kernel
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()



if __name__ == '__main__':
    end = datetime.now(timezone.utc) - timedelta(days=100)
    begin = end - timedelta(days=100)
    tickers=["MSFT", "AAPL", "TSLA", "IBM"]
    tickers.sort()
    params = Params(begin, end, tickers, "SVM", [Weight(WeightId())])
    df_api = yf.download(params.underlying_tickers, params.start_ts, params.end_ts, group_by="ticker", interval="1h")
    'List_Quote = Data_json.convert_data(df_api)
    'print(List_Quote)

    """
    -appel de l'API Yahoo et récupération des données de produits financiers à dates fixées dans une liste de quote
    """

    # Testing
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt
    for i in range(len(tickers)):
      df['H-L'] = List_Quote[i].high - List_Quote[i].low
      df_api['O-C'] = df_api['Close'] - df_api['Open']

    df_api['ma_5'] = df_api['Close'].rolling(window=5).mean()
    df_api['ma_10'] = df_api['Close'].rolling(window=10).mean()
   
    df_api['EWMA_12'] = df_api['Close'].ewm(span=12).mean()

    df_api['std_5'] = df_api['Close'].rolling(window=5).std()
    df_api['std_10'] = df_api['Close'].rolling(window=10).std()

    df_api['Price_Rise'] = np.where(df_api['Close'].shift(-1) > df_api['Close'], 1, 0)

    df_api = df_api.dropna()

    X = df_api.iloc[:, 4:-1]
    y = df_api.iloc[:, -1]

 #print(X_train, X_test, y_train, y_test)
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


    split = int(len(df_api) * 0.7)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler() #Fonction sklearn

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    clf = SVM()
    clf.fit(X_train, y_train)


    print(clf.fit)
    predictions = clf.predict(X_train)
    print(predictions)
    print(clf.w, clf.b)
    SVM.visualize_svm()

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)


    clf = SVM()


    clf.fit(X, y) #fit(X, y[, sample_weight]) : Fit the SVM model according to the given training data.
    predictions = clf.predict(X) # predict(X) : Perform regression on samples in X.

    print(predictions)
    print(clf.w, clf.b)

    SVM.visualize_svm()


