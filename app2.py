import customtkinter as ctk
from ttkthemes import ThemedTk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import re

def biseccion(f, a, b, tol=1e-6, max_iter=5):
    if f(a) * f(b) > 0:
        return ["El intervalo no contiene una raíz."]
    
    resultados = []
    for i in range(max_iter):
        c = (a + b) / 2
        resultados.append((i+1, a, b, c, f(c)))
        if abs(f(c)) < tol:
            return resultados
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return resultados

def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    resultados = []
    for i in range(max_iter):
        x1 = g(x0)
        resultados.append((i + 1, x0, x1))
        
        if abs(x1 - x0) < tol:
            return resultados
        
        x0 = x1
    
    return resultados

def newton(f, df, x0, tol=1e-6, max_iter=5):
    resultados = []
    for i in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        resultados.append((i+1, x0, x1))
        if abs(x1 - x0) < tol:
            return resultados
        x0 = x1
    return resultados

def integracion_trapecio(f, a, b, n):
    h = (b - a) / n
    suma = f(a) + f(b)
    resultados = [(0, a, f(a)), (n, b, f(b))]
    for i in range(1, n):
        xi = a + i * h
        fi = f(xi)
        suma += 2 * fi
        resultados.append((i, xi, fi))
    integral = (h / 2) * suma
    resultados.append(("Integral", integral))
    return resultados

def diferenciacion(f, x, h=1e-5):
    deltaX = h / 2
    L1 = (f(x + deltaX) - f(x)) / deltaX
    L2 = (f(x) - f(x - deltaX)) / deltaX
    L3 = (f(x + deltaX) - f(x - deltaX)) / (2 * deltaX)
    return [("L1", L1), ("L2", L2), ("L3", L3)]

def lagrange(x, y, xp):
    n = len(x)
    yp = 0
    resultados = []
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (xp - x[j]) / (x[i] - x[j])
        yp += y[i] * L
        resultados.append((i, L, yp))
    resultados.append(("Interpolación", yp))
    return resultados

def gauss_jacobi(A, b, x0, tol=1e-6, max_iter=5):
    n = len(b)
    x = x0.copy()
    resultados = []
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            suma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i][i]
        resultados.append((k+1, x_new.copy()))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return resultados
        x = x_new
    return resultados

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=5):
    n = len(b)
    x = x0.copy()
    resultados = []
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            suma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i][i]
        resultados.append((k+1, x_new.copy()))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return resultados
        x = x_new
    return resultados

def mostrar_resultado(resultados):
    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados")
    text = tk.Text(ventana_resultados, wrap=tk.WORD)
    text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    for resultado in resultados:
        text.insert(tk.END, str(resultado) + "\n")

def preprocess_function(func_str):
    func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
    func_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2')
    func_str = func_str.replace('^', '**')
    return func_str

def crear_ventana_metodo(titulo, labels, ejecutar_func):
    ventana = tk.Toplevel(root)
    ventana.title(titulo)
    
    entries = []
    for i, label in enumerate(labels):
        ctk.CTkLabel(ventana, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="e")
        entry = ctk.CTkEntry(ventana)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries.append(entry)
    
    ctk.CTkButton(ventana, text="Calcular", command=lambda: ejecutar_func(entries)).grid(row=len(labels), column=0, columnspan=2, pady=10)
    ventana.grid_columnconfigure(0, weight=1)
    ventana.grid_columnconfigure(1, weight=1)

def ejecutar_biseccion(entries):
    funcion = preprocess_function(entries[0].get())
    f = lambda x: eval(funcion)
    a = float(entries[1].get())
    b = float(entries[2].get())
    tol = float(entries[3].get() or 1e-6)
    max_iter = int(entries[4].get() or 5)
    resultados = biseccion(f, a, b, tol, max_iter)
    mostrar_resultado(resultados)

def ejecutar_punto_fijo(entries):
    funcion = preprocess_function(entries[0].get())
    g = lambda x: eval(funcion)
    x0 = float(entries[1].get())
    tol = float(entries[2].get() or 1e-6)
    max_iter = int(entries[3].get() or 5)
    resultados = punto_fijo(g, x0, tol, max_iter)
    mostrar_resultado(resultados)

def ejecutar_newton(entries):
    funcion = preprocess_function(entries[0].get())
    derivada = preprocess_function(entries[1].get())
    f = lambda x: eval(funcion)
    df = lambda x: eval(derivada)
    x0 = float(entries[2].get())
    tol = float(entries[3].get() or 1e-6)
    max_iter = int(entries[4].get() or 5)
    resultados = newton(f, df, x0, tol, max_iter)
    mostrar_resultado(resultados)

def ejecutar_integracion_trapecio(entries):
    funcion = preprocess_function(entries[0].get())
    f = lambda x: eval(funcion)
    a = float(entries[1].get())
    b = float(entries[2].get())
    n = int(entries[3].get())
    resultados = integracion_trapecio(f, a, b, n)
    mostrar_resultado(resultados)

def ejecutar_diferenciacion(entries):
    funcion = preprocess_function(entries[0].get())
    f = lambda x: eval(funcion)
    x0 = float(entries[1].get())
    h = float(entries[2].get() or 1e-5)
    resultados = diferenciacion(f, x0, h)
    mostrar_resultado(resultados)

def crear_ventana_lagrange():
    ventana = tk.Toplevel(root)
    ventana.title("Interpolación de Lagrange - Seleccionar Número de Puntos")

    ctk.CTkLabel(ventana, text="Número de puntos n:").grid(row=0, column=0)
    entry_n = ctk.CTkEntry(ventana)
    entry_n.grid(row=0, column=1)

    def siguiente():
        n = int(entry_n.get())
        ventana.destroy()
        crear_ventana_lagrange_valores(n)

    ctk.CTkButton(ventana, text="Siguiente", command=siguiente).grid(row=1, column=0, columnspan=2, pady=10)

def crear_ventana_lagrange_valores(n):
    ventana = tk.Toplevel(root)
    ventana.title("Interpolación de Lagrange - Ingresar Valores")

    entries_x = []
    entries_y = []

    for i in range(n):
        ctk.CTkLabel(ventana, text=f"x{i}:").grid(row=i, column=0, padx=10, pady=5, sticky="e")
        entry_x = ctk.CTkEntry(ventana, width=5)
        entry_x.grid(row=i, column=1, padx=10, pady=5, sticky="w")
        entries_x.append(entry_x)

        ctk.CTkLabel(ventana, text=f"y{i}:").grid(row=i, column=2, padx=10, pady=5, sticky="e")
        entry_y = ctk.CTkEntry(ventana, width=5)
        entry_y.grid(row=i, column=3, padx=10, pady=5, sticky="w")
        entries_y.append(entry_y)

    ctk.CTkLabel(ventana, text="xp:").grid(row=n, column=0, padx=10, pady=5, sticky="e")
    entry_xp = ctk.CTkEntry(ventana, width=5)
    entry_xp.grid(row=n, column=1, padx=10, pady=5, sticky="w")

    def ejecutar():
        x = [float(entry.get()) for entry in entries_x]
        y = [float(entry.get()) for entry in entries_y]
        xp = float(entry_xp.get())
        resultados = lagrange(x, y, xp)
        mostrar_resultado(resultados)

    ctk.CTkButton(ventana, text="Calcular", command=ejecutar).grid(row=n+1, column=0, columnspan=4, pady=10)

def crear_ventana_gauss_jacobi():
    ventana = tk.Toplevel(root)
    ventana.title("Método de Gauss-Jacobi - Seleccionar Tamaño de Matriz")

    ctk.CTkLabel(ventana, text="Número de filas:").grid(row=0, column=0)
    entry_filas = ctk.CTkEntry(ventana)
    entry_filas.grid(row=0, column=1)

    ctk.CTkLabel(ventana, text="Número de columnas:").grid(row=1, column=0)
    entry_columnas = ctk.CTkEntry(ventana)
    entry_columnas.grid(row=1, column=1)

    def siguiente():
        filas = int(entry_filas.get())
        columnas = int(entry_columnas.get())
        ventana.destroy()
        crear_ventana_gauss_jacobi_valores(filas, columnas)

    ctk.CTkButton(ventana, text="Siguiente", command=siguiente).grid(row=2, column=0, columnspan=2, pady=10)

def crear_ventana_gauss_jacobi_valores(filas, columnas):
    ventana = tk.Toplevel(root)
    ventana.title("Método de Gauss-Jacobi - Ingresar Valores")

    entries_A = []
    for i in range(filas):
        row_entries = []
        for j in range(columnas):
            entry = ctk.CTkEntry(ventana, width=5)
            entry.grid(row=i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        entries_A.append(row_entries)

    ctk.CTkLabel(ventana, text="=").grid(row=0, column=columnas, rowspan=filas, padx=10, pady=5)
    
    entries_b = []
    for i in range(filas):
        entry = ctk.CTkEntry(ventana, width=5)
        entry.grid(row=i, column=columnas+1, padx=5, pady=5)
        entries_b.append(entry)

    entries_x0 = []
    for i in range(columnas):
        ctk.CTkLabel(ventana, text=f"x{i}0:").grid(row=filas, column=i, padx=10, pady=5, sticky="e")
        entry = ctk.CTkEntry(ventana, width=5)
        entry.grid(row=filas+1, column=i, padx=10, pady=5, sticky="w")
        entries_x0.append(entry)

    ctk.CTkLabel(ventana, text="tol:").grid(row=filas+2, column=0, padx=10, pady=5, sticky="e")
    entry_tol = ctk.CTkEntry(ventana)
    entry_tol.grid(row=filas+2, column=1, padx=10, pady=5, sticky="w")

    ctk.CTkLabel(ventana, text="max_iter:").grid(row=filas+3, column=0, padx=10, pady=5, sticky="e")
    entry_max_iter = ctk.CTkEntry(ventana)
    entry_max_iter.grid(row=filas+3, column=1, padx=10, pady=5, sticky="w")

    def ejecutar():
        A = np.array([[float(entry.get()) for entry in row] for row in entries_A])
        b = np.array([float(entry.get()) for entry in entries_b])
        x0 = np.array([float(entry.get()) for entry in entries_x0])
        tol = float(entry_tol.get() or 1e-6)
        max_iter = int(entry_max_iter.get() or 5)
        resultados = gauss_jacobi(A, b, x0, tol, max_iter)
        mostrar_resultado(resultados)

    ctk.CTkButton(ventana, text="Calcular", command=ejecutar).grid(row=filas+4, column=0, columnspan=columnas+2, pady=10)

def crear_ventana_gauss_seidel():
    ventana = tk.Toplevel(root)
    ventana.title("Método de Gauss-Seidel - Seleccionar Tamaño de Matriz")

    ctk.CTkLabel(ventana, text="Número de filas:").grid(row=0, column=0)
    entry_filas = ctk.CTkEntry(ventana)
    entry_filas.grid(row=0, column=1)

    ctk.CTkLabel(ventana, text="Número de columnas:").grid(row=1, column=0)
    entry_columnas = ctk.CTkEntry(ventana)
    entry_columnas.grid(row=1, column=1)

    def siguiente():
        filas = int(entry_filas.get())
        columnas = int(entry_columnas.get())
        ventana.destroy()
        crear_ventana_gauss_seidel_valores(filas, columnas)

    ctk.CTkButton(ventana, text="Siguiente", command=siguiente).grid(row=2, column=0, columnspan=2, pady=10)

def crear_ventana_gauss_seidel_valores(filas, columnas):
    ventana = tk.Toplevel(root)
    ventana.title("Método de Gauss-Seidel - Ingresar Valores")

    entries_A = []
    for i in range(filas):
        row_entries = []
        for j in range(columnas):
            entry = ctk.CTkEntry(ventana, width=5)
            entry.grid(row=i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        entries_A.append(row_entries)

    ctk.CTkLabel(ventana, text="=").grid(row=0, column=columnas, rowspan=filas, padx=10, pady=5)
    
    entries_b = []
    for i in range(filas):
        entry = ctk.CTkEntry(ventana, width=5)
        entry.grid(row=i, column=columnas+1, padx=5, pady=5)
        entries_b.append(entry)

    entries_x0 = []
    for i in range(columnas):
        ctk.CTkLabel(ventana, text=f"x{i}0:").grid(row=filas, column=i, padx=10, pady=5, sticky="e")
        entry = ctk.CTkEntry(ventana, width=5)
        entry.grid(row=filas+1, column=i, padx=10, pady=5, sticky="w")
        entries_x0.append(entry)

    ctk.CTkLabel(ventana, text="tol:").grid(row=filas+2, column=0, padx=10, pady=5, sticky="e")
    entry_tol = ctk.CTkEntry(ventana)
    entry_tol.grid(row=filas+2, column=1, padx=10, pady=5, sticky="w")

    ctk.CTkLabel(ventana, text="max_iter:").grid(row=filas+3, column=0, padx=10, pady=5, sticky="e")
    entry_max_iter = ctk.CTkEntry(ventana)
    entry_max_iter.grid(row=filas+3, column=1, padx=10, pady=5, sticky="w")

    def ejecutar():
        A = np.array([[float(entry.get()) for entry in row] for row in entries_A])
        b = np.array([float(entry.get()) for entry in entries_b])
        x0 = np.array([float(entry.get()) for entry in entries_x0])
        tol = float(entry_tol.get() or 1e-6)
        max_iter = int(entry_max_iter.get() or 5)
        resultados = gauss_seidel(A, b, x0, tol, max_iter)
        mostrar_resultado(resultados)

    tk.Button(ventana, text="Calcular", command=ejecutar).grid(row=filas+4, column=0, columnspan=columnas+2)

root = tk.Tk()
root.title("Métodos Numéricos")

tk.Button(root, text="Método de Bisección", command=lambda: crear_ventana_metodo(
    "Método de Bisección",
    ["Función f(x):", "a:", "b:", "tol:", "max_iter:"],
    ejecutar_biseccion
)).grid(row=0, column=0, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Método de Punto Fijo", command=lambda: crear_ventana_metodo(
    "Método de Punto Fijo",
    ["Función g(x):", "x0:", "tol:", "max_iter:"],
    ejecutar_punto_fijo
)).grid(row=0, column=1, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Método de Newton", command=lambda: crear_ventana_metodo(
    "Método de Newton",
    ["Función f(x):", "Derivada f'(x):", "x0:", "tol:", "max_iter:"],
    ejecutar_newton
)).grid(row=1, column=0, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Integración Numérica", command=lambda: crear_ventana_metodo(
    "Integración Numérica por el Método del Trapecio",
    ["Función f(x):", "a:", "b:", "n:"],
    ejecutar_integracion_trapecio
)).grid(row=1, column=1, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Diferenciación", command=lambda: crear_ventana_metodo(
    "Diferenciación Numérica",
    ["Función f(x):", "x0:", "h:"],
    ejecutar_diferenciacion
)).grid(row=2, column=0, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Interpolación de Lagrange", command=crear_ventana_lagrange).grid(row=2, column=1, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Método de Gauss-Jacobi", command=crear_ventana_gauss_jacobi).grid(row=3, column=0, padx=10, pady=10, sticky="ew")

tk.Button(root, text="Método de Gauss-Seidel", command=crear_ventana_gauss_seidel).grid(row=3, column=1, padx=10, pady=10, sticky="ew")

root.mainloop()