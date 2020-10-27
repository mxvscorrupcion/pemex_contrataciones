import pandas as pd
import numpy as np
from pathlib import Path
from .utils import homologar_razon_social


def cargar_no_localizados(path: str):
    df = pd.read_csv(path, encoding='latin-1', usecols=['RFC', 'RAZÓN SOCIAL'])
    df = df.rename(columns={'RAZÓN SOCIAL': 'razon_social'})
    rs = df.razon_social.fillna('').astype(str).str.upper()
    rs = homologar_razon_social(rs).replace('', np.nan)
    df = df.assign(razon_social=rs)
    df = df.drop_duplicates()
    return df


def cargar_padron_proveedores(path: str):
    names = {
        'DENOMINACIÓN O RAZÓN SOCIAL DEL PROVEEDOR O CONTRATISTA': 'razon_social',
        'RFC DE LA PERSONA FÍSICA O MORAL CON HOMOCLAVE INCLUIDA': 'RFC',
    }
    dfs = [
        pd.read_csv(p, encoding='latin-1', skiprows=3, usecols=names.keys())
        for p in Path(path).glob('*/*/*/*/32*csv')
    ]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.rename(columns=names)
    rs = df.razon_social.fillna('').astype(str).str.upper()
    rs = homologar_razon_social(rs).replace('', np.nan)
    df = df.assign(razon_social=rs)
    df = df.drop_duplicates()
    return df


def cargar_lista_contribuyentes_69b(path):
    """Función que carga y limpia la tabla del Listado completo
    de contribueyentes artículo 69 B.
    http://omawww.sat.gob.mx/cifras_sat/Paginas/datos/vinculo.html?page=ListCompleta69B.html
    Parameters
    ----------
    path: str
        Ruta del archivo descargado de la página
    Returns
    -------
        Tabla con el nombre de la empresa limpia y el RFC
    """
    df = pd.read_csv(
        path,
        encoding="iso-8859-1",
        skiprows=2,
        usecols=["RFC", "Nombre del Contribuyente", 'Situación del contribuyente'],
    )
    df = df.rename(
        columns={"Nombre del Contribuyente": "razon_social",
                 'Situación del contribuyente': 'situacion_contribuyente'}
    )
    rs = df.razon_social.fillna('').astype(str).str.upper().str.strip()
    rs = homologar_razon_social(rs).replace('', np.nan)
    df = df.assign(razon_social=rs)
    df = df.dropna().drop_duplicates()
    return df


def cargar_particulares_sancionados(path):
    names = {'nombre_razon_social': 'razon_social', 'rfc': 'RFC'}
    df = pd.read_json(path)
    df = df.rename(columns=names)
    df = df.loc[:, ['razon_social', 'RFC']]
    rs = df.razon_social.fillna('').astype(str).str.upper().str.strip()
    rs = homologar_razon_social(rs).replace('', np.nan)
    df = df.assign(razon_social=rs)
    df = df.drop_duplicates()
    return df


def cargar_proveedores_sancionados(path: str):
    names = {'PROVEEDOR O CONTRATISTA': 'razon_social'}
    df = pd.read_csv(path, encoding='latin-1', usecols=['PROVEEDOR O CONTRATISTA'])
    df = df.rename(columns=names)
    rs = df.razon_social.fillna('').astype(str).str.upper().str.strip()
    rs = homologar_razon_social(rs).replace('', np.nan)
    df = df.assign(razon_social=rs)
    df = df.dropna().drop_duplicates()
    return df


# def cargar_listado_proveedores(path: str) -> pd.DataFrame:
#     names = {
#         'DENOMINACIÓN O RAZÓN SOCIAL DEL PROVEEDOR O CONTRATISTA': 'razon_social',
#         'RFC DE LA PERSONA FÍSICA O MORAL CON HOMOCLAVE INCLUIDA': 'RFC',
#     }
#     df = pd.read_csv(path, usecols=names.keys())
#     df = df.rename(columns=names)
#     rs = df.razon_social.fillna('').astype(str).str.upper()
#     rs = homologar_razon_social(rs).replace('', np.nan)
#     df = df.assign(razon_social=rs)
#     df = df.drop_duplicates()
#     return df
