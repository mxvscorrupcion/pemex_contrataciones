"""Funciones para calcular los features a nivel proveedor"""
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


def features_binarios_proveedores(
        procedimientos: pd.DataFrame,
        ofertas: pd.DataFrame,
        proveedores_fantasma: pd.DataFrame,
        proveedores_no_localizados: pd.DataFrame,
        proveedores_sancionados: pd.DataFrame,
        particulares_sancionados: pd.DataFrame,
        padron_proveedores: pd.DataFrame,
        n_proveedores: int = 1089) -> pd.DataFrame:
    dfs = [
        empresa_no_en_padron_proveedores(procedimientos, padron_proveedores),
        reportada_como_empresa_fantasma(procedimientos, proveedores_fantasma),
        empresa_no_localizada_sat(procedimientos, proveedores_no_localizados),
        tasa_exito_proveedor(procedimientos, ofertas, 0.5),
        empresa_creada_recientemente(procedimientos),
        market_share_por_contratos(procedimientos),
        market_share_por_monto(procedimientos),
        participacion_conjunta_sospechosa(procedimientos, ofertas, 5, 0.5, 0.5),
        proveedores_y_particulares_sancionados(
            procedimientos, proveedores_sancionados, particulares_sancionados
        ),
    ]
    dfs = [df.set_index('razon_social_simple') for df in dfs]
    if not all([df.shape[0] == n_proveedores for df in dfs]):
        m = ('Existen shapes que no son iguales al número de proveedores indicado. '
             f'Los shapes son los siguientes: {[df.shape[0] for df in dfs]}')
        raise ValueError(m)
    df_features_proveedores = (pd.concat(dfs, axis=1, ignore_index=False)
                               .reset_index()
                               .rename(columns={'index': 'razon_social_simple'}))
    cols = [
        'tasa_exito',
        'fecha_creacion_rfc',
        # market share monto
        'share_empresa_monto_pep_adquisiciones',
        'share_empresa_monto_pep_arrendamientos',
        'share_empresa_monto_pep_obra_pública',
        'share_empresa_monto_pep_servicios',
        'share_empresa_monto_plo_adquisiciones',
        'share_empresa_monto_plo_servicios',
        'share_empresa_monto_pmx_adquisiciones',
        'share_empresa_monto_pmx_servicios',
        'share_empresa_monto_pti_obra_pública',
        'share_empresa_monto_ptri_adquisiciones',
        'share_empresa_monto_ptri_arrendamientos',
        'share_empresa_monto_ptri_obra_pública',
        'share_empresa_monto_ptri_servicios',
        # market share contratos
        'share_empresa_contratos_pep_adquisiciones',
        'share_empresa_contratos_pep_arrendamientos',
        'share_empresa_contratos_pep_obra_pública',
        'share_empresa_contratos_pep_servicios',
        'share_empresa_contratos_plo_adquisiciones',
        'share_empresa_contratos_plo_servicios',
        'share_empresa_contratos_pmx_adquisiciones',
        'share_empresa_contratos_pmx_servicios',
        'share_empresa_contratos_pti_obra_pública',
        'share_empresa_contratos_ptri_adquisiciones',
        'share_empresa_contratos_ptri_arrendamientos',
        'share_empresa_contratos_ptri_obra_pública',
        'share_empresa_contratos_ptri_servicios',
    ]
    df_features_proveedores = df_features_proveedores.drop(cols, axis=1)
    return df_features_proveedores


def empresa_creada_recientemente(procedimientos: pd.DataFrame,
                                 fecha_max: str = '2018-01-01') -> pd.DataFrame:
    # FIXME: algunas empresas tienen más de un RFC. Puede ser error de captura
    # FIXME: se reporta con 0 a las empresas que tienen nulo su RFC. Debe ser Nan
    # feature 13
    empresas = (procedimientos.loc[:, ['razon_social_simple']]
                .dropna().drop_duplicates())
    cols_group = ['razon_social_simple', 'RFC', 'fecha_creacion_rfc']
    data = (procedimientos.groupby(cols_group).numero_contrato.nunique()
            .reset_index()
            .sort_values('numero_contrato', ascending=False))
    data = (data.loc[~data.fecha_creacion_rfc.isna()]
            .drop_duplicates('razon_social_simple')
            .drop(['RFC', 'numero_contrato'], axis=1)
            .reset_index(drop=True))
    reciente = (data.fecha_creacion_rfc >= pd.Timestamp(fecha_max)).astype(int)
    data = data.assign(empresa_reciente=reciente)
    feature = pd.merge(empresas, data, 'left', on='razon_social_simple')
    return feature


def tasa_exito_proveedor(
        procedimientos: pd.DataFrame,
        ofertas: pd.DataFrame,
        threshold: float) -> pd.DataFrame:
    # feature 19
    rs = 'razon_social_simple'
    empresas = procedimientos.loc[:, [rs]].dropna().drop_duplicates()
    cond = procedimientos.tipo_procedimiento != 'Adjudicación directa'
    df_ganadores = (procedimientos.loc[cond]
                    .groupby('razon_social_simple').numero_contrato.nunique()
                    .reset_index()
                    .rename(columns={'numero_contrato': 'contratos_ganados'}))
    df_ganadores = df_ganadores.loc[df_ganadores.razon_social_simple != 'S']
    ganadores = df_ganadores.razon_social_simple.unique()
    ofertas = ofertas.loc[:, ['numero_contrato', rs]].drop_duplicates()
    participaciones = defaultdict(int)
    for contrato, participantes in ofertas.groupby('numero_contrato'):
        conjunto = participantes[rs].values
        conjunto = itertools.chain.from_iterable(e.split('/') for e in conjunto)
        conjunto = [e.strip() for e in conjunto]
        for g, p in itertools.product(ganadores, conjunto):
            cond = g == p
            # cond = g in p
            # caso con regex
            if cond:
                participaciones[g] += 1
    df_participaciones = pd.DataFrame.from_dict(
        participaciones, orient='index', columns=['participaciones']
    )
    df_participaciones = df_participaciones.reset_index().rename(columns={'index': rs})
    df_join = pd.merge(df_ganadores, df_participaciones, on=rs, how='left')
    df_join = df_join.loc[~df_join.participaciones.isna()]
    # no puede ser que haya más contratos ganados que participaciones
    df_join = df_join.loc[df_join.contratos_ganados <= df_join.participaciones]
    # nos quedamos con los ganadores con más de tres participaciones
    df_join = df_join.loc[df_join.participaciones >= 3]
    tasa = df_join.contratos_ganados.divide(df_join.participaciones)
    df_join = df_join.drop(['participaciones', 'contratos_ganados'], axis=1)
    tasa_alta = (tasa > threshold).astype(int)
    df_join = df_join.assign(tasa_exito=tasa, tasa_exito_alta=tasa_alta)
    df_join = pd.merge(empresas, df_join, on=rs, how='left')
    return df_join


def empresa_no_en_padron_proveedores(
        df: pd.DataFrame,
        listado_proveedores: pd.DataFrame) -> pd.DataFrame:
    # feature 8
    listado_proveedores_nombre = listado_proveedores['razon_social'].unique()
    listado_proveedores_rfc = listado_proveedores['RFC'].unique()
    data = df.copy()
    data = data.loc[:, ['RFC', 'razon_social_simple']]
    # Realizar revisión por nombre y por RFC
    not_in_contractors_list_1 = np.where(
        data.razon_social_simple.isin(listado_proveedores_nombre), 0, 1
    )
    not_in_contractors_list_2 = np.where(
        data.RFC.isin(listado_proveedores_rfc), 0, 1
    )
    data = data.assign(
        not_in_contractors_list_1=not_in_contractors_list_1,
        not_in_contractors_list_2=not_in_contractors_list_2,
    )
    # Agrupar por razon social simple
    data_grouped = (data.groupby('razon_social_simple')
                    .agg({'not_in_contractors_list_1': 'sum',
                          'not_in_contractors_list_2': 'sum'})
                    .reset_index())
    no_en_padron = (data_grouped.not_in_contractors_list_1 +
                    data_grouped.not_in_contractors_list_2)
    no_en_padron = np.where(no_en_padron > 0, 1, 0)
    data_grouped = data_grouped.assign(no_en_padron_proveedores=no_en_padron)
    feature = data_grouped.loc[:, ['razon_social_simple', 'no_en_padron_proveedores']]
    return feature


def reportada_como_empresa_fantasma(df: pd.DataFrame,
                                    listado_fantasmas: pd.DataFrame) -> pd.DataFrame:
    # feature 9
    estatus_fantasma = {'Definitivo', 'Presunto'}
    # Filtrar base de fantasma para definitivos y presuntos
    fantasmas = listado_fantasmas.copy()
    fantasmas = fantasmas[fantasmas['situacion_contribuyente'].isin(estatus_fantasma)]
    # Crear listas
    fantasma_rfc = fantasmas['RFC'].unique()
    fantasma_nombre = fantasmas['razon_social'].unique()
    # Unir con base de contratos
    data = df.copy()
    data = data.loc[:, ['RFC', 'razon_social_simple']]
    is_phantom_1 = np.where(data['razon_social_simple'].isin(fantasma_nombre), 1, 0)
    is_phantom_2 = np.where(data['RFC'].isin(fantasma_rfc), 1, 0)
    data = data.assign(is_phantom_1=is_phantom_1, is_phantom_2=is_phantom_2)
    # Agrupar por razón social simple
    data_grouped = (data.groupby('razon_social_simple')
                    .agg({'is_phantom_1': 'sum',
                          'is_phantom_2': 'sum'})
                    .reset_index())
    # Valor 1 si el proveedor se encuentra por cualquiera de los 2 métodos
    es_fantasma = data_grouped.is_phantom_1 + data_grouped.is_phantom_2
    es_fantasma = np.where(es_fantasma > 0, 1, 0)
    data_grouped = data_grouped.assign(es_empresa_fantasma=es_fantasma)
    feature = data_grouped.loc[:, ['razon_social_simple', 'es_empresa_fantasma']]
    return feature


def empresa_no_localizada_sat(df: pd.DataFrame,
                              no_localizados: pd.DataFrame) -> pd.DataFrame:
    # feature 10
    # Lista de proveedores no localizados
    no_localizados_rfc = no_localizados.RFC
    no_localizados_nombre = no_localizados.razon_social
    # Unir con base de contratos
    data = df.copy()
    data = data.loc[:, ['RFC', 'razon_social_simple']]
    is_not_found_1 = np.where(
        data.razon_social_simple.isin(no_localizados_nombre), 1, 0
    )
    is_not_found_2 = np.where(data.RFC.isin(no_localizados_rfc), 1, 0)
    data = data.assign(is_not_found_1=is_not_found_1, is_not_found_2=is_not_found_2)
    # Agrupar por razón social simple
    data_grouped = (data.groupby('razon_social_simple')
                    .agg({'is_not_found_1': 'sum',
                          'is_not_found_2': 'sum'})
                    .reset_index())
    # Valor 1 si el proveedor se encuentra por cualquiera de los 2 métodos
    no_localizada = data_grouped.is_not_found_1 + data_grouped.is_not_found_2
    no_localizada = np.where(no_localizada > 0, 1, 0)
    data_grouped = data_grouped.assign(empresa_no_localizada=no_localizada)
    feature = data_grouped.loc[:, ['razon_social_simple', 'empresa_no_localizada']]
    return feature


def proveedores_y_particulares_sancionados(df: pd.DataFrame,
                                           proveedores_sancionados: pd.DataFrame,
                                           particulares_sancionados: pd.DataFrame):
    # TODO: de donde sale la de particulares sancionados?
    # feature 11
    # Listado de proveedores sancionados
    listado_sancionados_1 = proveedores_sancionados['razon_social'].unique()
    listado_sancionados_2 = particulares_sancionados['razon_social'].unique()
    listado_sancionados_3 = particulares_sancionados['RFC'].unique()
    # Cruzar listas con listado de proveedores
    # listado_proveedores_1 = listado_proveedores.loc[
    #     listado_proveedores['razon_social'].isin(listado_sancionados_1)
    # ]
    # listado_proveedores_2 = listado_proveedores.loc[
    #     listado_proveedores['razon_social'].isin(listado_sancionados_2)
    # ]
    # listado_proveedores_1_rfc = listado_proveedores_1['RFC']
    # listado_proveedores_1_nombre = listado_proveedores_1['razon_social']
    # listado_proveedores_2_rfc = listado_proveedores_2['RFC']
    # listado_proveedores_2_nombre = listado_proveedores_2['razon_social']
    # Unir con base de contratos
    data = df.copy()
    data = data.loc[:, ['RFC', 'razon_social_simple']]
    data['sanctioned_1'] = np.where(
        data['razon_social_simple'].isin(listado_sancionados_1), 1, 0
    )
    data['sanctioned_2'] = np.where(
        data['razon_social_simple'].isin(listado_sancionados_2), 1, 0
    )
    # data['sanctioned_3'] = np.where(
    #     data['razon_social_simple'].isin(listado_proveedores_1_nombre), 1, 0
    # )
    # data['sanctioned_4'] = np.where(
    #     data['razon_social_simple'].isin(listado_proveedores_2_nombre), 1, 0
    # )
    # data['sanctioned_5'] = np.where(data['RFC'].isin(listado_proveedores_1_rfc), 1, 0)
    # data['sanctioned_6'] = np.where(data['RFC'].isin(listado_proveedores_2_rfc), 1, 0)

    data['sanctioned_3'] = np.where(
        data['RFC'].isin(listado_sancionados_3), 1, 0
    )

    # Agrupar por razón social simple
    data_grouped = (data.groupby('razon_social_simple')
                    .agg({'sanctioned_1': 'sum',
                          'sanctioned_2': 'sum',
                          'sanctioned_3': 'sum',
                          # 'sanctioned_4': 'sum',
                          # 'sanctioned_5': 'sum',
                          # 'sanctioned_6': 'sum',
                          }).reset_index())

    # Valor 1 si el proveedor se encuentra por cualquiera de los 6 métodos
    cond = ((data_grouped['sanctioned_1'] >= 1) |
            (data_grouped['sanctioned_2'] >= 1) |
            (data_grouped['sanctioned_3'] >= 1))
    # (data_grouped['sanctioned_4'] >= 1) |
    # (data_grouped['sanctioned_5'] >= 1) |
    # (data_grouped['sanctioned_6'] >= 1))
    data_grouped['empresa_sancionada'] = np.where(cond, 1, 0)
    feature = data_grouped[['razon_social_simple', 'empresa_sancionada']]
    return feature


def market_share_por_monto(procedimientos: pd.DataFrame,
                           cuantil_min=0.9) -> pd.DataFrame:
    # feature 17
    data = procedimientos.copy()
    empresas = data.loc[:, ['razon_social_simple']].dropna().drop_duplicates()
    # Agrupar por empresa, materia y proveedor
    cols_group = ['empresa_productiva', 'materia', 'razon_social_simple']
    data_grouped = (data.groupby(cols_group)
                    .agg({'monto': 'sum', 'ID': 'nunique'}).reset_index())
    # Crear columna de monto total
    totales = (data.groupby(['empresa_productiva', 'materia'])
               .agg({'monto': 'sum', 'ID': 'nunique'}).reset_index())
    totales = totales.rename(
        columns={'monto': 'monto_total', 'ID': 'contratos_total'}
    )
    # Unir tablas
    data_final = pd.merge(
        data_grouped, totales, how='left', on=['empresa_productiva', 'materia']
    )
    # se eliminan los grupos con pocos contratos
    data_final = data_final.loc[data_final.contratos_total >= 6]
    # Obtener shares
    data_final['share_empresa'] = data_final['monto'] / data_final['monto_total']
    dfs = []
    for (e, m), df_group in data_final.groupby(['empresa_productiva', 'materia']):
        cond = (data_final.empresa_productiva == e) & (data_final.materia == m)
        cuantil = data_final.loc[cond].share_empresa.dropna().quantile(cuantil_min)
        df_group = df_group.assign(
            supera_umbral=(df_group.share_empresa > cuantil).astype(int)
        )
        col_share_tipo = f'share_empresa_monto_{e}_{"_".join(m.split())}'.lower()
        df_group.loc[:, col_share_tipo] = df_group.share_empresa
        dfs.append(df_group)
    data_final = pd.concat(dfs, axis=0, ignore_index=True)
    cols_to_sum = [c for c in data_final.columns if c.startswith('share_empresa_')]
    cols_to_sum.append('supera_umbral')
    feature = (data_final.groupby('razon_social_simple', as_index=False)
               .agg({c: 'sum' for c in cols_to_sum}))
    feature['market_share_monto_riesgoso'] = np.where(
        feature['supera_umbral'] >= 1, 1, 0
    )
    feature = pd.merge(empresas, feature, 'left', on='razon_social_simple')
    feature = feature.drop('supera_umbral', axis=1)
    return feature


def market_share_por_contratos(df: pd.DataFrame, cuantil_min=0.9):
    # feature 18
    data = df.copy()
    empresas = data.loc[:, ['razon_social_simple']].dropna().drop_duplicates()
    # Agrupar por empresa, materia y proveedor
    cols_group = ['empresa_productiva', 'materia', 'razon_social_simple']
    data_grouped = (data.groupby(cols_group)
                    .agg({'ID': 'nunique'})
                    .reset_index())
    # Crear columna de contratos totales
    totales = (data.groupby(['empresa_productiva', 'materia'])
               .agg({'ID': 'nunique'})
               .reset_index())
    totales = totales.rename(columns={'ID': 'contratos_total'})
    # Unir tablas
    data_final = pd.merge(
        data_grouped, totales, 'left', on=['empresa_productiva', 'materia']
    )
    # se eliminan los grupos con pocos contratos
    data_final = data_final.loc[data_final.contratos_total >= 6]
    # Obtener shares
    data_final['share_empresa'] = data_final['ID'] / data_final['contratos_total']
    dfs = []
    for (e, m), df_group in data_final.groupby(['empresa_productiva', 'materia']):
        cond = (data_final.empresa_productiva == e) & (data_final.materia == m)
        cuantil = data_final.loc[cond].share_empresa.dropna().quantile(cuantil_min)
        df_group = df_group.assign(
            supera_umbral=(df_group.share_empresa > cuantil).astype(int)
        )
        col_share_tipo = f'share_empresa_contratos_{e}_{"_".join(m.split())}'.lower()
        df_group.loc[:, col_share_tipo] = df_group.share_empresa
        dfs.append(df_group)
    data_final = pd.concat(dfs, axis=0, ignore_index=True)
    cols_to_sum = [c for c in data_final.columns if c.startswith('share_empresa_')]
    cols_to_sum.append('supera_umbral')
    feature = (data_final.groupby('razon_social_simple', as_index=False)
               .agg({c: 'sum' for c in cols_to_sum}))
    feature['market_share_contratos_riesgoso'] = np.where(
        feature['supera_umbral'] >= 1, 1, 0
    )
    feature = feature.drop('supera_umbral', axis=1)
    feature = pd.merge(empresas, feature, 'left', on='razon_social_simple')
    return feature


def participacion_conjunta_sospechosa(
        procedimientos: pd.DataFrame,
        participantes: pd.DataFrame,
        participaciones_min: int,
        ratio_part_conjunta_min: float,
        tasa_exito_min: float) -> pd.DataFrame:
    rs = 'razon_social_simple'
    participantes = participantes.loc[:, ['numero_contrato', rs]].drop_duplicates()
    cond = procedimientos.tipo_procedimiento != 'Adjudicación directa'
    # FIXME: podemos usar los numero_contrato para sacar las
    #  participaciones también. En lugar de solo usar razon social
    ganadores = (procedimientos.loc[cond, [rs, 'numero_contrato']]
                 .groupby(rs).numero_contrato.nunique()
                 .reset_index()
                 .rename(columns={'numero_contrato': 'contratos_ganados'}))
    contratos_to_participantes = participantes.groupby('numero_contrato')[rs].unique()
    # se genera una mapeo entre el número de contrato y la lista de sus participantes
    contratos_to_participantes = contratos_to_participantes.map(
        lambda arr: list(itertools.chain.from_iterable(e.split('/') for e in arr))
    ).to_dict()
    contratos_to_participantes = {
        k: [e.strip() for e in v]
        for k, v in contratos_to_participantes.items()
    }
    contratos_to_participantes = {
        k: [e for e in v if e != '']
        for k, v in contratos_to_participantes.items()
    }
    # se calcula la matriz de empresa a empresa con el numero
    # de participaciones
    conteos = {}
    for g in ganadores[rs]:
        conjuntos_participantes = []
        for empresas in contratos_to_participantes.values():
            if g in empresas:
                conjuntos_participantes.append(empresas)
        if len(conjuntos_participantes):
            conteos[g] = Counter(
                itertools.chain.from_iterable(conjuntos_participantes)
            )
    matriz = (pd.DataFrame.from_dict(conteos, orient='index')
              .fillna(0).reset_index())
    # dataframe: empresa_ganadora, empresa_participante, numero de participaciones
    tidy = matriz.melt(
        id_vars='index', var_name='empresa_participante', value_name='participaciones'
    )
    tidy = tidy.rename(columns={'index': 'empresa_ganadora'})
    # escogemos a las empresas con suficientes participaciones
    empresas_selected = tidy.loc[tidy.empresa_ganadora == tidy.empresa_participante]
    empresas_selected = empresas_selected.loc[
        empresas_selected.participaciones >= participaciones_min,
        'empresa_ganadora'
    ]
    empresas_selected = empresas_selected.to_numpy()
    # Se filtran los registros con 0 participaciones. Además solo usamos
    # los registros con las empresas seleccionadas
    tidy = tidy.loc[tidy.empresa_ganadora.isin(empresas_selected)]
    tidy = tidy.loc[tidy.participaciones > 0]
    # se calcula la relación de participación conjunta entre la
    # empresa ganadora y las otras participantes
    empresas_sospechosas = []
    for e in empresas_selected:
        info_participaciones = tidy.loc[tidy.empresa_ganadora == e]
        # numero participaciones de la empresa ganadora
        total_participaciones = info_participaciones.loc[
            info_participaciones.empresa_participante == e, 'participaciones'
        ].values[0]
        ratios_participacion_conjunta = info_participaciones.loc[
            info_participaciones.empresa_participante != e,
            'participaciones'
        ].divide(total_participaciones)
        if (ratios_participacion_conjunta >= ratio_part_conjunta_min).any():
            record = {rs: e, 'participaciones': total_participaciones}
            empresas_sospechosas.append(record)
    df_empresas_sosp = pd.DataFrame(empresas_sospechosas)
    df_empresas_sosp = pd.merge(df_empresas_sosp, ganadores, 'inner', on=rs)
    ratio_exito = df_empresas_sosp.contratos_ganados.divide(
        df_empresas_sosp.participaciones
    )
    cond_tasa = ratio_exito >= tasa_exito_min
    df_empresas_sosp = df_empresas_sosp.loc[cond_tasa]
    df_empresas_sosp = df_empresas_sosp.assign(
        misma_competencia_y_tasa_exito_alta=1
    )
    feature = procedimientos.loc[:, [rs]].dropna().drop_duplicates()
    feature = feature.loc[~feature[rs].isin(df_empresas_sosp[rs])]
    feature = pd.concat([df_empresas_sosp, feature], axis=0, ignore_index=True)
    feature_cols = [rs, 'misma_competencia_y_tasa_exito_alta']
    feature = feature.loc[:, feature_cols].fillna(0)
    return feature
