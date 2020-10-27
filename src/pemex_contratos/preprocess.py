import re
import numpy as np
import pandas as pd
from typing import List, Optional
from unicodedata import normalize, lookup
from datetime import datetime


REGEX_LIST: List[str] = [
    # SA DE CV
    r"[S]\s?[A]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SAPI DE CV
    r"[S]\s?[A]\s?[P]\s?[I]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SAB DE CV
    r"[S]\s?[A]\s?[B]\s+[D]\s?[E]\s+[C]\s?[V]",
    # S DE RL DE CV
    r"[S]\s?[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # S DE RL
    r"[S]\s?[D]\s?[E]\s+[R]\s?[L]",
    # SC DE RL DE CV
    r"[S]\s?[C]\s+[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SCP DE RL DE CV
    r"[S]\s?[C]\s?[P]\s+[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SPR DE RL DE CV
    r"[S]\s?[P]\s?[R]\s+[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # PR DE RL DE CV
    r"[P]\s?[R]\s+[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SC DE C DE RL DE CV
    r"[S]\s?[C]\s+[D]\s?[E]\s+[C]\s+[D]\s?[E]\s+[R]\s?[L]\s+[D]\s?[E]\s+[C]\s?[V]",
    # SPR DE RL
    r"[S]\s?[P]\s?[R]\s+[D]\s?[E]\s+[R]\s?[L]",
    # S DE PR DE RL
    r"[S]\s+[D]\s?[E]\s+[P]\s?[R]\s+[D]\s?[E]\s+[R]\s?[L]",
]


def read_lista_contribuyentes_69b(path):
    """
    Función que carga y limpia la tabla del Listado completo
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
        usecols=["RFC", "Nombre del Contribuyente"],
    )
    df = df.rename(columns={"Nombre del Contribuyente": "empresa_fantasma"})
    df = df.assign(
        empresa_fantasma=clean_razon_social(df.empresa_fantasma.str.strip().str.upper())
    )
    df = df.dropna().drop_duplicates()
    return df


def extraer_numero_iniciativa(numero_evento: str) -> Optional[int]:
    """Algunos numeros de evento traen la iniciativa a la
    que pertenecen, esta función intenta extraer el ID

    Returns
    -------
    int or None
        ID de iniciative o None si no existe un numero con las
        caracteristicas buscadas
    """
    tokens = re.split(r"\W+|_", numero_evento)
    iniciativa = None
    for token in tokens:
        if len(token) == 5 and token.isnumeric():
            iniciativa = int(token)
            return iniciativa
    return iniciativa


# TODO: add test
def fecha_creacion_from_rfc(rfc: pd.Series) -> pd.Series:
    """
    De la serie extrae los primeros 6 digitos para formar
    la fecha de creacion del RFC

    Returns
    -------
    Regresa una serie con fechas y Nats si fue posible extraer
    la fecha
    """
    pattern = re.compile(r"\d{6}")
    rfc = rfc.fillna("")
    # personas morales len(rfc) == 12
    # personas fisicas len(rfc) == 13
    rfc[rfc.str.len() < 12] = ""
    fechas = rfc.map(lambda string: pattern.findall(string))
    fechas = fechas.map(lambda l: l[0] if len(l) > 0 else None)
    years = fechas.str.slice(0, 2).astype(float)
    fechas_20 = pd.Series(data="20", index=fechas.index).str.cat(fechas)
    fechas_19 = pd.Series(data="19", index=fechas.index).str.cat(fechas)
    cond = (years >= 0) & (years <= 20)
    fechas = np.where(cond, fechas_20, fechas_19)
    # Algunas fechas no son válidas porque
    # son RFCs/identificadores de otros paises
    # o esta vacio el campo o está mal el dato
    fechas = pd.to_datetime(fechas, format="%Y%m%d", errors="coerce")
    return fechas


def read_eventos(path: str) -> pd.DataFrame:
    """Carga y limpia la tabla que viene en la pagina del siscep.
    Es la misma funcion para los 3 tipos de contrataciones"""
    df = (
        pd.read_csv(path, parse_dates=["Publicado"], dayfirst=True)
        .drop("Documentos", axis=1)
        .sort_values("Publicado", ascending=True)
        .reset_index(drop=True)
    )
    names = {
        "No. Evento": "num_evento",
        "Descripción": "descripcion",
        "Tipo de evento": "tipo_evento",
        "Tipo de suministro": "tipo_iniciativa",
        "Publicado": "publicado",
    }
    df = df.rename(columns=names)
    df = df.assign(iniciativa=df["num_evento"].map(extraer_numero_iniciativa))
    return df


def read_proveedores(path) -> pd.DataFrame:
    """Carga y limpia la lista de proveedores que esta en la
    página del siscep"""
    df = pd.read_excel(path, dtype={"Año de Registro": float, "Código Postal": str})
    names = {
        "Año de Registro": "año_registro",
        "Nombre ó Razón Social": "razon_social",
        "Representante Legal": "representante_legal",
        "Entidad Federativa": "entidad",
    }
    df = df.rename(columns=names)
    razon_social = clean_razon_social(df["razon_social"])
    df = df.assign(
        razon_social=razon_social,
        representante_legal=df["representante_legal"].str.upper(),
    )
    return df


def remove_expression(regex, series: pd.Series):
    pattern = re.compile(regex)
    new_series = series.map(lambda s: pattern.sub("", s).strip())
    return new_series


def remove_accents(s: str) -> str:
    """
    Quita los acentos del texto. Es decir,
    transforma 'compañía' a 'compañia'
    https://gist.github.com/j4mie/557354
    """
    accent = lookup("COMBINING ACUTE ACCENT")
    chars = [c for c in normalize("NFD", s) if c != accent]
    return normalize("NFC", "".join(chars))


def remove_ending_chars(s: str, ending: str):
    """Hay palabras que no se terminaron de eliminar con las
    expresiones regulares, con esta funcion se eliminan"""
    pattern = f" {ending}"
    n = len(re.findall(pattern, s))
    # if only one SA and it is the last one
    if n == 1 and s.endswith(pattern):
        # remove ending
        s = " ".join(s.split(pattern)[:-1])
    return s


def clean_razon_social(razon_social: pd.Series) -> pd.Series:
    """Normaliza los nombres de las empresas"""
    regular_expressions = REGEX_LIST
    nombre = razon_social.str.replace(".", "")
    nombre = nombre.str.replace(",", "")
    nombre = nombre.map(remove_accents)
    for regex in regular_expressions:
        nombre = remove_expression(regex, nombre)
    # Existen los casos ' SA' que no se filtran con las regex,
    known_endings = ["SC", "SA", "INC", "LLC", "SAPI"]
    for ending in known_endings:
        nombre = nombre.map(lambda s: remove_ending_chars(s, ending))
    # chamge '' to Nan
    nombre = nombre.replace(r"", np.nan, regex=True)
    return nombre


def read_adjudicaciones(path: str, path_scrapped_table: str) -> pd.DataFrame:
    """Carga y limpia la tabla de adjudicaciones creado por los capturistas."""
    cols = [
        "folder_id",
        "Monto Mínimo",
        "Monto Máximo",
        "Moneda",
        "IVA",
        "Razon Social",
        "Contrato adjudicado",
        "Signatario (PEMEX)",
        "NO ADJUDICADO",
        "OBSERVACIONES",
    ]
    names = {
        "Signatario (PEMEX)": "signatario_pemex",
        "Razon Social": "empresa_ganadora",
        "Monto Mínimo": "monto_minimo",
        "Monto Máximo": "monto_maximo",
        "NO ADJUDICADO": "Resultado",
    }
    df = pd.read_excel(path, usecols=cols)
    df = df.rename(columns=names)
    df = df.assign(
        # num_evento=df.num_evento.fillna(method="ffill"),
        folder_id=df.folder_id.fillna(method="ffill"),
    )
    # Rellenar proveedores faltantes con la sig condicion
    df = df.groupby(["folder_id"], as_index=False).fillna(method="ffill")
    # con_algun_monto = ~(df.monto_minimo.isna()) & (df.monto_maximo.isna())
    # eventos_con_prov_faltante = df.loc[
    #     con_algun_monto & df.proveedor.isna(), "num_evento"
    # ].unique()
    # relleno_provs = df.loc[
    #     df.num_evento.isin(eventos_con_prov_faltante), "proveedor"
    # ].fillna(method="ffill")
    # df.loc[relleno_provs.index, "proveedor"] = relleno_provs
    empresa_ganadora = df["empresa_ganadora"].fillna("").str.upper()
    empresa_ganadora = clean_razon_social(empresa_ganadora)
    df.loc[:, "empresa_ganadora"] = empresa_ganadora
    # Los montos minimos y maximos se llenan de forma complementaria
    # (sucede cuando no existe un rango)
    df = df.assign(monto_minimo=df.monto_minimo.fillna(df.monto_maximo))
    df = df.assign(monto_maximo=df.monto_maximo.fillna(df.monto_minimo))
    # add the missing data from the scrapped table
    df_eventos = read_eventos(path_scrapped_table)
    df = pd.merge(df, df_eventos, on=["folder_id"], how="left")
    df = df.rename(columns={"empresa": "empresa_pemex"})
    return df


def read_invitaciones(path: str, path_scrapped_table: str) -> pd.DataFrame:
    """Carga y limpia la table de invitaciones generado por los
    capturistas."""
    cols = [
        "No. Evento",
        "folder_id",
        "Monto Mínimo",
        "Monto Máximo",
        "Moneda",
        "IVA",
        "Contrato adjudicado",
        "Signatario (PEMEX)",
        "NO ADJUDICADO",
        "Empresa 1",
        "Empresa 2",
        "Empresa 3",
        "Empresa 4",
        "Empresa 5",
        "Empresa 6",
        "Empresa 7",
        "Empresa Ganadora",
        "OBSERVACIONES",
    ]
    names = {
        "Signatario (PEMEX)": "signatario_pemex",
        "No. Evento": "num_evento",
        "Monto Mínimo": "monto_minimo",
        "Monto Máximo": "monto_maximo",
        "Empresa 1": "empresa_1",
        "Empresa 2": "empresa_2",
        "Empresa 3": "empresa_3",
        "Empresa 4": "empresa_4",
        "Empresa 5": "empresa_5",
        "Empresa 6": "empresa_6",
        "Empresa 7": "empresa_7",
        "Empresa Ganadora": "empresa_ganadora",
        "NO ADJUDICADO": "no_adjudicado",
    }
    na_values = [
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
    ]
    df = pd.read_excel(
        path,
        usecols=cols,
        dtype={"Monto Máximo": float, "Monto Mínimo": float},
        na_values=na_values,
        keep_default_na=False,
    )
    df = df.rename(columns=names)
    df = df.assign(
        num_evento=df.num_evento.fillna(method="ffill"),
        folder_id=df.folder_id.fillna(method="ffill"),
    )
    for i in [1, 2, 3, 4, 5, 6, 7, "ganadora"]:
        empresa = df[f"empresa_{i}"].fillna("").str.upper()
        empresa = clean_razon_social(empresa)
        df.loc[:, f"empresa_{i}"] = empresa
    df = df.assign(monto_minimo=df.monto_minimo.fillna(df.monto_maximo))
    df = df.assign(monto_maximo=df.monto_maximo.fillna(df.monto_minimo))
    df_eventos = read_eventos(path_scrapped_table)
    df = pd.merge(df, df_eventos, on=["num_evento", "folder_id"], how="left")
    return df


def read_concursos_abiertos(path, path_scrapped_table):
    cols = [
        "folder_id",
        "Monto Mínimo",
        "Monto Máximo",
        "Moneda",
        "IVA",
        "Resultado ",
        "Singatario PEMEX",
        "Empresa ganadora",
        "Q&A",
        "Propuestas",
        "Fallo",
        "Observaciones",
        "EMPRESA 1",
        "EMPRESA 2",
        "EMPRESA 3",
        "EMPRESA 4",
        "EMPRESA 5",
    ]
    names = {
        "Monto Mínimo": "monto_minimo",
        "Monto Máximo": "monto_maximo",
        "Resultado ": "Resultado",
        "Singatario PEMEX": "signatario_pemex",
        "Empresa ganadora": "empresa_ganadora",
        "EMPRESA 1": "empresa_1",
        "EMPRESA 2": "empresa_2",
        "EMPRESA 3": "empresa_3",
        "EMPRESA 4": "empresa_4",
        "EMPRESA 5": "empresa_5",
        "Q&A": "Preguntas_y_respuestas",
    }
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns=names)
    m_min = df.monto_minimo.str.replace("$", "").str.replace(",", "").astype(float)
    m_max = df.monto_maximo.str.replace("$", "").str.replace(",", "").astype(float)
    df = df.assign(
        monto_minimo=m_min,
        monto_maximo=m_max,
        folder_id=df.folder_id.fillna(method="ffill"),
    )
    df = df.loc[~df.folder_id.str.contains("PTI")]
    df = df.groupby("folder_id", as_index=False).fillna(method="ffill")
    fallo = df.Fallo.replace("NM", np.nan)
    propuestas = df.Propuestas.replace("NM", np.nan)
    preguntas_y_respuestas = df.Preguntas_y_respuestas.replace("NM", np.nan)
    preguntas_y_respuestas = preguntas_y_respuestas.replace("NA ", np.nan)

    fallo = homologar_fechas_concursos(fallo)
    propuestas = homologar_fechas_concursos(propuestas)
    preguntas_y_respuestas = homologar_fechas_concursos(preguntas_y_respuestas)
    df = df.assign(
        Fallo=fallo,
        Propuestas=propuestas,
        Preguntas_y_respuestas=preguntas_y_respuestas,
    )
    for i in [1, 2, 3, 4, 5, "ganadora"]:
        empresa = df[f"empresa_{i}"].fillna("").str.upper()
        empresa = clean_razon_social(empresa)
        df.loc[:, f"empresa_{i}"] = empresa
    # Los montos minimos y maximos se llenan de forma complementaria
    # (sucede cuando no existe un rango)
    df = df.assign(monto_minimo=df.monto_minimo.fillna(df.monto_maximo))
    df = df.assign(monto_maximo=df.monto_maximo.fillna(df.monto_minimo))
    # se limpia columna 'Resultado'
    resultado = df.Resultado.replace("AJUDICADA", "ADJUDICADA")
    resultado = resultado.replace("ADJUDICADA ", "ADJUDICADA")
    resultado = resultado.replace("ADJUICADA", "ADJUDICADA")
    df = df.assign(Resultado=resultado)
    cond = df.Resultado.isna() & ~df.empresa_ganadora.isna()
    df.loc[cond, "Resultado"] = "ADJUDICADA"
    # se limpia campo de Moneda
    moneda = df.Moneda.replace("MNX", "MXN").replace("MXN ", "MXN")
    df = df.assign(Moneda=moneda)
    # add the missing data from the scrapped table
    df_eventos = read_eventos(path_scrapped_table)
    df = pd.merge(df, df_eventos, on=["folder_id"], how="left")
    return df


def homologar_fechas_concursos(fechas: pd.Series) -> pd.Series:
    formato1 = pd.to_datetime(fechas, format="%Y-%m-%d", errors="coerce")
    formato2 = pd.to_datetime(fechas, format="%d/%m/%y", errors="coerce")
    result = formato1.fillna(formato2)
    return result


def desagregar_consorcios(df: pd.DataFrame) -> pd.DataFrame:
    cond_consorcios = (~df.empresa_ganadora.isna()) & df.empresa_ganadora.str.contains(
        "/"
    )
    consorcios = df.loc[cond_consorcios]
    df_sin_consorcios = df.loc[~cond_consorcios]
    dfs = []
    for index, row in consorcios.iterrows():
        monto_min = row.montos_minimos_mxn
        monto_max = row.montos_maximos_mxn
        empresas = row.empresa_ganadora.split("/")
        empresas = [e.replace("(PROPUESTA CONJUNTA)", "") for e in empresas]
        empresas = [e.replace("(PROPUESTA CONJUNTA", "") for e in empresas]
        empresas = [e.strip() for e in empresas]
        n_empresas = len(empresas)
        monto_min = monto_min / n_empresas
        monto_max = monto_max / n_empresas
        new_rows = []
        for empresa in empresas:
            new_row = dict(row)
            new_row["empresa_ganadora"] = empresa
            new_row["montos_minimos_mxn"] = monto_min
            new_row["montos_maximos_mxn"] = monto_max
            new_row["consorcio"] = True
            new_rows.append(new_row)
        df_consorcio = pd.DataFrame(new_rows)
        dfs.append(df_consorcio)
    df_consorcios = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    df_consorcios = df_consorcios.assign(
        empresa_ganadora=clean_razon_social(df_consorcios.empresa_ganadora)
    )
    df_final = pd.concat(
        [df_sin_consorcios, df_consorcios], axis=0, ignore_index=True, sort=False
    )
    # se ordenan por folder if
    df_final = df_final.sort_values(["folder_id", "publicado"])
    return df_final


def asignar_montos_en_pesos(df: pd.DataFrame, tipos_cambio) -> pd.DataFrame:
    montos_minimos_mxn = []
    montos_maximos_mxn = []
    cols = ["publicado", "monto_minimo", "monto_maximo", "Moneda_asumida"]
    for row in df[cols].itertuples():
        moneda = row.Moneda_asumida
        if moneda:
            factor = tipos_cambio[datetime.date(row.publicado)][moneda]
            minimo = row.monto_minimo * factor
            maximo = row.monto_maximo * factor
        else:
            minimo = np.nan
            maximo = np.nan
        montos_minimos_mxn.append(minimo)
        montos_maximos_mxn.append(maximo)
    df = df.assign(
        montos_minimos_mxn=montos_minimos_mxn, montos_maximos_mxn=montos_maximos_mxn
    )
    return df


def process_invitaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Modifica la tabla de invitaciones para tener un formato
    num_evento,participante, estatus, monto_minimo, monto_maximo, moneda, iva
    """
    eventos = df.num_evento.dropna().value_counts()
    # FIXME: por el momento solo usamos los eventos que
    #  traen un solo regristro de la tabla de invitaciones
    eventos = eventos[eventos == 1].index.to_numpy()
    cols_participantes = [
        "empresa_1",
        "empresa_2",
        "empresa_3",
        "empresa_4",
        "empresa_5",
        "empresa_6",
        "empresa_7",
        "empresa_ganadora",
    ]
    cols_montos = ["monto_minimo", "monto_maximo", "moneda", "iva"]
    dataframes = []
    for evento in eventos:
        # montos
        df_montos = df.loc[df.num_evento == evento, cols_montos]
        # Participantes
        df_participantes = df.loc[df.num_evento == evento, cols_participantes].T
        df_participantes = df_participantes.rename(
            columns={c: "participante" for c in df_participantes.columns}
        )
        df_participantes = df_participantes.drop_duplicates("participante", keep="last")
        df_participantes = df_participantes.assign(estatus="perdedor")
        df_participantes.loc["empresa_ganadora", "estatus"] = "ganador"
        # asignamos informacion de montos a ganador
        df_participantes = df_participantes.assign(**{c: np.nan for c in cols_montos})
        # FIXME: esto no funciona cuando un evento tiene mas de un evento
        for c in cols_montos:
            df_participantes.loc["empresa_ganadora", c] = df_montos[c].unique()
        df_participantes = df_participantes.assign(num_evento=evento)
        dataframes.append(df_participantes)
    df_final = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)
    return df_final


def carga_tipos_de_cambio(path):
    """Carga y limpia la tabla de tipo de cambio del Banco de Mexico"""
    df = pd.read_csv(
        path,
        encoding="iso-8859-1",
        skiprows=17,
        parse_dates=["Fecha"],
        dayfirst=True,
        usecols=["Fecha", "SF46405", "SF46410"],
    ).rename(columns={"SF46405": "USD", "SF46410": "EUR"})
    rango = pd.date_range(df.Fecha.min(), df.Fecha.max(), freq="D")
    df = df.sort_values("Fecha").set_index("Fecha")
    df = df.reindex(rango)
    for col in df.columns:
        df.loc[:, col] = df[col].interpolate()
    df = df.assign(MXN=1)
    tipos = {datetime.date(k): v for k, v in df.to_dict(orient="index").items()}
    return tipos


def read_actualizacion_trimestral(path, skiprows: int, skipfooter: int) -> pd.DataFrame:
    """Carga la tabla del archivo de actualizacion trimestral de los
    programas anuales"""
    dates = [
        "FECHA PROGRAMADA ENTREGA SOLICITUD",
        "FECHA ESTIMADA DE FIRMA",
        "FECHA INICIO DE CONTRATO",
        "FECHA TERMINO DE CONTRATO",
    ]
    names = {
        "ID INICIATIVA": "iniciativa",
        "TOTAL MN": "total_mn",
        "PRIORIDAD": "prioridad",
        "PROYECTO ASOCIADO": "proyecto_asociado",
    }
    df = pd.read_excel(
        path,
        skiprows=skiprows,
        skipfooter=skipfooter,
        parse_dates=dates,
        dtype={"ID INICIATIVA": int},
    )
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    df = df.rename(columns=names)
    # remove trailing white spaces
    df = df.rename(columns={c: c.strip() for c in df.columns if isinstance(c, str)})
    df = df.rename(columns={"DESCRIPCION GENERAL": "descripcion_general"})
    for col in ["descripcion_general", "proyecto_asociado"]:
        texto = df[col].str.upper().str.strip().fillna("")
        texto = texto.map(remove_accents).replace("", np.nan)
        df.loc[:, col] = texto
    return df
