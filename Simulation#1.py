#This program is based in the class 03/07/26 HIC
#Simulation by computer 
import numpy as np

class AlgebraLineal:
    """
    Clase para manejar operaciones de álgebra lineal utilizando NumPy.
    No se utilizan frameworks de machine learning como scikit-learn.
    """
    def __init__(self):
        pass

    def transpuesta(self, matriz):
        return np.transpose(matriz)

    def inversa(self, matriz):
        return np.linalg.inv(matriz)

    def multiplicar(self, a, b):
        return np.dot(a, b)

class Regresion:
    """
    Clase para el modelo de regresión lineal/polimonial usando mínimos cuadrados.
    Atributos:
    - beta: coeficientes del modelo
    Métodos:
    - completar_beta(): calcula los coeficientes beta
    - print_re(): imprime el R-cuadrado (medida de ajuste)
    - predecir(x): predice valores para nuevas entradas x
    """
    def __init__(self, X, y, grado=1):
        """
        Inicializa el modelo.
        - X: array de valores independientes (univariado)
        - y: array de valores dependientes
        - grado: grado del polinomio (1 para lineal, 2 para cuadrático, etc.)
        """
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)  # Asegurar columna
        self.grado = grado
        self.beta = None
        self.algebra = AlgebraLineal()

    def completar_beta(self):
        """
        Calcula los coeficientes beta usando la fórmula de mínimos cuadrados:
        beta = (X^T X)^-1 X^T y
        """
        n = len(self.X)
        # Matriz de diseño: [1, x, x^2, ..., x^grado]
        diseño = np.ones((n, self.grado + 1))
        for i in range(1, self.grado + 1):
            diseño[:, i] = np.power(self.X, i)
        
        Xt = self.algebra.transpuesta(diseño)
        XtX = self.algebra.multiplicar(Xt, diseño)
        XtX_inv = self.algebra.inversa(XtX)
        Xty = self.algebra.multiplicar(Xt, self.y)
        self.beta = self.algebra.multiplicar(XtX_inv, Xty)

    def print_re(self):
        """
        Calcula e imprime el R-cuadrado (R^2) como medida de ajuste del modelo.
        R^2 = 1 - (SS_res / SS_tot)
        """
        if self.beta is None:
            raise ValueError("Primero debe completar beta.")
        
        y_pred = self.predecir(self.X)
        ss_res = np.sum((self.y.flatten() - y_pred.flatten()) ** 2)
        ss_tot = np.sum((self.y.flatten() - np.mean(self.y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        print(f"R-cuadrado (R^2): {r2:.4f}")

    def predecir(self, x):
        """
        Predice valores y para nuevas entradas x.
        - x: array o lista de valores a predecir
        """
        if self.beta is None:
            raise ValueError("Primero debe completar beta.")
        
        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1)
        n = len(x)
        # Matriz de diseño para predicciones
        diseño = np.ones((n, self.grado + 1))
        for i in range(1, self.grado + 1):
            diseño[:, i] = np.power(x, i)
        
        return self.algebra.multiplicar(diseño, self.beta).flatten()

class LSR:
    """
    Clase Least Squares Regressor (LSR) como interfaz para mostrar datos predictivos.
    Se conecta a la clase Regresion y realiza predicciones.
    """
    def __init__(self, X, y, grado=1):
        """
        Inicializa LSR con datos y crea el modelo de regresión.
        """
        self.regresion = Regresion(X, y, grado)
        self.regresion.completar_beta()
        print("Modelo entrenado. Coeficientes beta:")
        print(self.regresion.beta.flatten())
        self.regresion.print_re()

    def mostrar_predicciones(self, nuevos_x):
        """
        Muestra al menos 5 predicciones para nuevos valores x.
        - nuevos_x: lista o array con al menos 5 valores para predecir
        """
        if len(nuevos_x) < 5:
            raise ValueError("Debe proporcionar al menos 5 valores para predecir.")
        
        predicciones = self.regresion.predecir(nuevos_x)
        print("\nPredicciones:")
        for xi, pred in zip(nuevos_x, predicciones):
            print(f"Para x = {xi}, prediccion y = {pred:.4f}")

# Ejemplo de uso basado en las notas (regresión polinomial)
# Datos de ejemplo: supongamos una relacion no lineal simple y = 2x^2 + 3x + 1 + ruido
np.random.seed(42)
X_ejemplo = np.array([1, 2, 3, 4, 5, 6])
y_ejemplo = 2 * X_ejemplo**2 + 3 * X_ejemplo + 1 + np.random.normal(0, 1, len(X_ejemplo))

# Crear instancia de LSR con grado 2 (cuadratico, como en las notas)
lsr = LSR(X_ejemplo, y_ejemplo, grado=2)

# Hacer al menos 5 predicciones para nuevos x
nuevos_x = [7, 8, 9, 10, 11]
lsr.mostrar_predicciones(nuevos_x)