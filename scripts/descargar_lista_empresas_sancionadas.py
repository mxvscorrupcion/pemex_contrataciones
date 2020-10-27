"""Script que descarga la lista de sancionados del siguiente link:
https://directoriosancionados.funcionpublica.gob.mx/SanFicTec/jsp/Ficha_Tecnica/SancionadosN.htm
"""
import helium
import gazpacho
import pandas as pd
from pemex_contratos.preprocess import clean_razon_social

if __name__ == "__main__":
    # inputs del script
    url = "https://directoriosancionados.funcionpublica.gob.mx/SanFicTec/jsp/Ficha_Tecnica/SancionadosN.htm"
    file_path = "../data/processed/lista_empresas_sancionadas.csv"
    # se inicia el headless browser para bajar la informacion
    driver = helium.start_firefox(url, headless=True)
    helium.Alert().accept()
    helium.click("TODOS")
    soup = gazpacho.Soup(driver.page_source)
    helium.kill_browser()
    # procesamiento de los datos
    tabla_sancionados = soup.find("table")
    elementos = tabla_sancionados.find("strong")
    nombres_sancionados = [e.text.strip().upper() for e in elementos]
    df_sancionados = pd.DataFrame(nombres_sancionados, columns=["empresa_sancionada"])
    df_sancionados = df_sancionados.drop_duplicates()
    df_sancionados = (
        df_sancionados.assign(
            empresa_sancionada=clean_razon_social(df_sancionados.empresa_sancionada)
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_sancionados.to_csv(file_path, encoding="utf-8", quoting=1, index=False)
