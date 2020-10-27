# pemex_contrataciones
Análisis exploratorio sobre los procesos de contratación de Pemex

# Datos
Las fuentes de datos usados en el análisis son los siguientes:
* [Contrataciones y padrón de proveedores de PEMEX](https://www.pemex.com/transparencia/Paginas/obligaciones-transparencia.aspx)
* [Listado de contribuyentes artículo 69B](http://omawww.sat.gob.mx/cifras_sat/Paginas/datos/vinculo.html?page=ListCompleta69B.html)
* [Proveedores no localizados](http://omawww.sat.gob.mx/cifras_sat/Documents/No%20localizados.csv)
* [Proveedores y contratistas sancionados](https://datos.gob.mx/busca/dataset/proveedores-y-contratistas-sancionados/resource/737d9402-d9ff-4662-b9c9-02770e303637)
* [Particulares sancionados](https://plataformadigitalnacional.org/sancionados)

Además se creo un programa para descargar la información del 
[Sistema de Contrataciones Electrónicas Pemex](https://www.pemex.com/procura/procedimientos-de-contratacion/contratacion/Paginas/default.aspx) 
que se encuentra en la carpeta de scripts.

## Estructura de carpetas

TODO:
Es probable que los datos se actualicen y puede que la estructura cambie

# Código
Hay dos programas principales en la carpeta de scripts y ambos asumen que tienes 
la estructura de carpetas.

* El script ```calculcar_features.py``` calcula los valores reales de los features 
para algunos de los conceptos utilizados y regresa los features binarios utilizados 
para calcular el score de riesgo 
* El script ```calculcar_score_riesgo.py``` calcula los features binarios de los
conceptos y también regresa el score de riesgo.

Ambos regresan una tabla que se guardará en el folder especificado en cada uno de 
ellos.

