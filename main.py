# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable

def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą bisekcji.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    fa = f(a)
    fb = f(b)
    
    
    if fa * fb > 0:
        return None
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        
        
        if abs(fc) < epsilon or (b - a)/2 < epsilon:
            return c, i
        
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    
    return (a + b)/2, max_iter



def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] 
    metodą siecznych.

    Args:
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iters (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    fa = f(a)
    fb = f(b)
    x1 = a
    x0 = b
    i = 0
    if fa*fb > 0:
        return None
    while abs(x0-x1) > epsilon and i < max_iters:
        i += 1
        x1 = x0
        x0 = a - fa*(b-a)/(fb-fa)
        f0 = f(x0)
        if abs(f0) < epsilon:
            return x0, i
        if fa*f0 < 0:
            b = x0
            fb = f0
        else:
            a = x0
            fa = f0
    return x0, i
def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    """Funkcja obliczająca wartość iloazu różnicowego w punkcie x dla zadanej 
    funkcji f(x).

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        x (int | float): Argument funkcji.
        h (int | float): Krok różnicy wykorzystywanej do wyliczenia ilorazu 
            różnicowego.

    Returns:
        (float): Wartość ilorazu różnicowego.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if h == 0:
        return None
    return (f(x + h) - f(x)) / h

def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        df (Callable[[float], float]): Pierwsza pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        ddf (Callable[[float], float]): Druga pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    x = a if f(a) * ddf(a) > 0 else b
    fx = f(x)

    for iteration in range(1, max_iter + 1):
        dfx = df(x)
        if dfx == 0:
            return None

        x_new = x - fx / dfx
        fx_new = f(x_new)

        if abs(fx_new) < epsilon:
            return np.float64(x_new), iteration

        x = x_new
        fx = fx_new

    return np.float64(x), max_iter