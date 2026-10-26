<h1 align="center">Strateu, EU fork of Strata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

<p align="center"><b>Ejecuta un modelo de IA de 125 mil millones de parámetros en tu propio PC gaming</b><br>
Tarjeta gráfica NVIDIA o AMD (12 GB o más) · Windows o Linux · libre y de código abierto</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Un jardín con una pagoda de vóxeles que escribió el modelo de Strata, funcionando en el navegador"></a><br>
<sub>Un jardín con una pagoda de vóxeles, hecho con un solo prompt en una RTX 5070 con Strata (IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

Strata ejecuta **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** en un PC normal. Es un
modelo de IA grande e inteligente que normalmente necesita un servidor. Conversa, escribe código, entiende imágenes
y funciona con tus aplicaciones y agentes de programación. Nada sale de tu PC. También ejecuta **[Kolibri 1](docs/KOLIBRI1.md)** (el MoE alemán e inglés de 78B de Aleph Alpha, 3,46B activos) con 56 GB de RAM: `./setup.sh --family kolibri` (Q4_K_M, el único tamaño que el setup ofrece por ahora).

## Sobre este fork

**strateu** es un fork medido de [Niko1221/Strata](https://github.com/Niko1221/Strata) centrado en el motor
**Kolibri 1** (`src/kolibri`). Una semana de mediciones dejó el port CUDA **determinista bit a bit** (el prefill
pasa racecheck, diferencia 0.0 entre dos ejecuciones) y **verificado contra llama.cpp por divergencia KL**
(0,00132 en 109 pasos de respuesta, top-1 0,982, oracle del camino simple en BITWISE). En una tarjeta de 16 GB, medido con reloj (dos ejecuciones cada uno): unos **119-120 tokens/s** de prefill tras un
prompt de 32K, **30 tokens/s** de generación corto y **13-14 tokens/s** tras 32K
(remeido tras una auditoría: cifras antes derivadas de un contador). la aguja de 100K se recupera en 4 de 5 casos y las dos
desviaciones quedan registradas (la propia referencia llama.cpp es inestable a esa longitud). Las compuertas y la
configuración de velocidad están en `tools/kolibri/`; cada número proviene de dos ejecuciones con la carga de la
máquina indicada. La evidencia completa: [el informe de la campaña](docs/strateu-campaign-report.md) y los registros bajo `docs/superpowers/plans/`.

## ¿Qué velocidad tiene?

Lo medimos en dos PC gaming normales. Un token equivale más o menos a ¾ de una palabra.

- **Escribe respuestas:** lo rápido que aparece la respuesta en un chat corto. 60 tokens por segundo es más rápido de lo que puedes leer.
- **Lee tu prompt:** lo rápido que procesa lo que le envías (aquí, un documento, código o historial de chat de 32K tokens).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Tamaño | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 con el motor 0.1.36, las demás filas con 0.1.26 (respuestas de 4K, prompts de 32K). Las tablas
completas están en [DETAILS.md](docs/DETAILS.md#speed-measured). Una tarjeta con más VRAM es más rápida: una
RTX 3090 (24 GB) debería escribir unos 100-140 tokens por segundo. Chats largos y otras tarjetas:
[velocidad de cada modelo](docs/MODELS.md#how-fast-is-each-size), [resultados de la comunidad](docs/COMMUNITY_BENCHMARKS.md).

Este fork también midió **[Kolibri 1](docs/KOLIBRI1.md)** de Aleph Alpha (Q4_K_M) en una **Tesla T4 (16 GB)** con
755 GB de RAM para el pool de expertos:

| Prompt | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **corto (512 tokens)** | 30 tokens/s (según afinidad NUMA) | 156-160 tokens/s (`PF_PARTIAL` opcional: 180-181, en caliente 202) |
| **32K tokens** | 13-14 tokens/s (según afinidad NUMA) | 119-120 tokens/s |
| **100K tokens** | — | 97 tokens/s |

El mismo archivo en la Radeon integrada de un Ryzen AI Max+ 395 (Radeon 8060S, Strix Halo, `gfx1151`, HIP sobre ROCm 7.14.1) con 123 GiB de RAM para el pool de expertos:

| Prompt | Escribe respuestas | Lee tu prompt |
| --- | ---: | ---: |
| **corto (512 tokens)** | 23,8 tokens/s | 168,5 tokens/s |
| **32K tokens** | 9,8 tokens/s | 114,0 tokens/s |

No se midió ningún prompt de 100K en la iGPU.

Dos ejecuciones por número con el binario determinista bit a bit, carga de la máquina indicada; la fila de 100K es
una única ejecución en frío. Esas filas se tomaron con 256K de contexto y una caché de expertos **dimensionada a
mano**. La versión actual dimensiona la caché sola y puede colocar los expertos según la frecuencia con la que el
router los elige:

- `--expert-slots auto` mide el espacio libre tras los pesos densos, el KV y el scratch, y tomó **75 slots/capa
  (8,49 GiB)** con **160K** de contexto, dejando 700 MiB libres para el runtime de CUDA. Imprime el tipo de cambio
  que implica: 1 slot = 114 MiB = 5.857 tokens de contexto, porque contexto y caché salen del mismo VRAM.
- `--expert-slots-hot` llena esos slots por frecuencia de selección del router en vez de por id de experto, y
  `--expert-slots-profile` conserva el conjunto aprendido entre reinicios. En esta T4 la tasa de acierto pasó de
  **21,4 %** (por id) a **33,4-34,2 %** en tres prompts distintos, **37,9 %** al repetir el mismo prompt y **31,4 %**
  en la primera petición de un arranque en frío sembrado desde un perfil guardado - para **24,4-26,3 tokens/s** de
  decodificación con sampling. Greedy es ~30 % más rápido y entra en bucle en este modelo
  (`Chudnov Chudnov Chudnov ...`), por eso el servidor hace sampling por defecto.
- El motor acepta **160K** tokens de contexto; la configuración del cliente declara 128K, por debajo.

El intercambio no cuesta nada en el flujo de tokens pero sí en tiempo: promover 57 expertos costó 6,5 GiB por PCIe en
~690 ms, pagados entre peticiones. La residencia mueve bytes, nunca los cambia - los ids generados fueron idénticos
en las ejecuciones por id, hot y auto. Especificación de la máquina y tablas completas:
[el informe de la campaña](docs/strateu-campaign-report.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata es gratis. Si funciona bien en tu PC, un café ayuda a seguir trabajando en él.</sub></p>

## Qué necesitas

| | |
| --- | --- |
| **Tarjeta gráfica** | **NVIDIA** GeForce serie RTX 20, 30, 40 o 50, o **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 o serie RX 6800 / 6900. Necesita **12 GB de VRAM o más**. |
| **RAM** | 32 GB o más. Tu RAM decide [qué modelo](#qué-modelo-elijo) cabe. Con 64 GB funcionan todos los tamaños. |
| **Disco** | Unos 80 GB libres. Usa un SSD si puedes: el primer arranque es mucho más rápido. |
| **Sistema** | Windows 10 / 11 o Linux, y un controlador gráfico actual de NVIDIA o AMD. |

El instalador prepara todo lo demás. Dos o tres tarjetas pueden repartirse el modelo ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, escrito y probado por miembros de la comunidad en sus propios equipos:

- **Tarjetas gráficas antiguas** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, compilado desde el código fuente en Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilado desde el código fuente en Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Procesadores antiguos sin AVX2**: funcionan, pero despacio. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

La lista completa: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Instalación

### Deja que tu IA lo instale

¿Usas un asistente de programación con IA (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Pega esto en él:

```text
Set up Strateu on this PC for me: https://github.com/gboddaer/strateu - follow docs/AI_SETUP.md in that repository.
```

Revisa tu tarjeta gráfica, tu RAM y tu disco, y elige el modelo que cabe. Luego lo instala, lo arranca y te explica
cómo conectar tus aplicaciones. Las herramientas de IA también pueden instalar, arrancar y detener Strata con su
[servidor MCP](docs/MCP_SERVER.md).

### O hazlo tú mismo

[Descarga Strateu](https://github.com/gboddaer/strateu/archive/refs/heads/main.zip) y descomprímelo (o usa `git clone`).
**Windows:** haz doble clic en **`START-HERE.bat`**. **Linux:** ejecuta **`./setup.sh`** en la carpeta de Strata.

**Mac (Apple Silicon, M1 a M5, macOS 15 o posterior):** instala una vez las Command Line Tools de Apple (`xcode-select --install` en Terminal; no hace falta Xcode). Luego abre Terminal en la carpeta de Strata y ejecuta **`./setup.sh`**. Compila el motor de Mac en tu Mac (unos 7 minutos), hace las mismas preguntas, descarga el modelo y lo arranca. La próxima vez, `./setup.sh` arranca enseguida. Qué tamaño cabe en tu Mac y qué velocidad tiene: [docs/MAC.md](docs/MAC.md).

Los pasos son los mismos para NVIDIA y AMD. El instalador detecta tu tarjeta y prepara el motor adecuado para ella.
Te hace unas pocas preguntas:

- qué modelo y qué tamaño,
- cuánto contexto (cuánto texto tiene presente el modelo),
- si debe entender imágenes.

Pulsa Enter cada vez para usar la respuesta recomendada. Después descarga el modelo (unos 70 GB) y lo arranca. Si la
descarga se detiene, vuelve a ejecutarlo: sigue donde lo dejó. Tu navegador abre la aplicación de Strata en
`http://127.0.0.1:8080`.

> **Mientras el modelo arranca, tu PC puede ir lento o dejar de responder durante 1-3 minutos** (más la primera vez).
> Strata carga 35-55 GB en tu RAM y reserva una parte para la tarjeta gráfica. Es normal. Espera y no cierres la
> ventana. La ventana muestra lo que está haciendo Strata.

**La próxima vez**, vuelve a ejecutar `START-HERE.bat` (o `./setup.sh`). Arranca enseguida y no descarga nada dos
veces. Cierra su ventana para detener el modelo. `UPDATE.bat` (`./update.sh`) actualiza Strata sin arrancarlo.
Actualizaciones, Docker, varias tarjetas, dónde se guardan los archivos y todas las opciones: [docs/INSTALL.md](docs/INSTALL.md).

## ¿Qué modelo elijo?

El instalador te recomienda uno según tu RAM. El mismo modelo viene en varios tamaños, más o menos comprimidos. Los
tamaños pequeños son más rápidos. Los grandes son algo más inteligentes.

| Tu RAM | Elige | Por qué |
| --- | --- | --- |
| **32 GB** | **Coder** | cabe en 32 GB y está hecho para programar (con una tarjeta de 24 GB también funcionan Q2_0 e IQ2_XS) |
| **48 GB** | **IQ2_XS** (o Q2_0, el más rápido) | los tamaños más grandes no caben |
| **64 GB** | **IQ2_XS** (recomendado), o IQ3_XXS / IQ3_S | caben todos los tamaños; IQ3_S es el mejor y el más lento |
| **96 GB o más** | **IQ3_S**, o el UD-IQ4_XS de Unsloth (~4 bits) | espacio para los tamaños más grandes con todo lo demás abierto |

- **[Coder](docs/MODELS.md#coder):** una versión para programar a la que se le quitó la mitad de los expertos. Alcanza
  el 91% de la puntuación SWE-bench Verified del modelo completo (medido por sus autores) y cabe en 32 GB de RAM. Es
  más flojo fuera del código, también con chino y otros textos CJK (#438). Para eso, elige Q2_0, IQ2_XS o IQ3_S, que
  conservan todos los expertos.
- **[Swift 1.5](docs/MODELS.md#swift-15):** un ajuste fino que piensa mucho menos tiempo antes de responder. Recibes
  la respuesta antes, con más o menos la misma calidad.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** la versión de ~4 bits de Unsloth, entre IQ3_S y
  UD-Q4_K_XL en calidad. Una descarga de 94 GB. Con menos de ~80 GB de RAM, Strata lee una parte desde el SSD
  mientras responde, así que ahí es más lento (un SSD NVMe ayuda).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL y Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):** de 4 a 8 bits, más cerca del modelo completo
  (93-95 % el mismo token siguiente que Q8_0). Rápidos cuando todos los expertos caben en la RAM (~88 GB para
  UD-Q4_K_XL hasta ~136 GB para Q8_0); con menos RAM, Strata lee una parte desde el SSD mientras responde, lo que es
  mucho más lento (7-8.5 tokens/s para UD-Q4_K_XL en un PC con 64 GB).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** se instala a mano. No está en
  el menú del instalador.

Tamaños, descargas y qué cabe dónde: [docs/MODELS.md](docs/MODELS.md). Para añadir otro modelo más adelante, ejecuta
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Cómo usarlo

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="La pestaña Monitor de la aplicación de Strata junto a un agente de programación"><br>
<sub>El <b>Monitor</b> de la aplicación de Strata (izquierda) mientras un agente de programación escribe el jardín de la pagoda del vídeo (derecha)</sub></p>

- **En el navegador:** abre `http://127.0.0.1:8080`. Tiene **Chat**, un **Monitor** en vivo del modelo y de tu
  GPU/CPU/RAM, y **About** con los ajustes y las direcciones.
- **Tus aplicaciones y agentes de programación:** añade un proveedor "compatible con OpenAI" con la URL base
  **`http://127.0.0.1:8080/v1`**. Sirve cualquier clave de API y cualquier nombre de modelo.
  - Aplicaciones que usan la API de Anthropic: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI y otras aplicaciones que usan la API Responses de OpenAI: `/v1/responses`
    ([configuración](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Razonamiento:** elige **off, low, medium o high** en el menú del chat o en el "reasoning effort" de tu aplicación.
  Off es lo más rápido. High es lo mejor para preguntas difíciles.
- **Imágenes:** responde que sí a "Images?" en la instalación. Luego haz clic en **Picture** en el chat, o adjunta
  imágenes en tu aplicación. Las tarjetas AMD leen imágenes en Linux a través del procesador; en Windows todavía no.
- **Desde tu móvil u otro PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Pon siempre una clave.
- **Una petición cada vez:** por defecto Strata responde una petición y las demás esperan. Para responder varias a la
  vez, pon `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). En una tarjeta de 12 GB, cada respuesta va más lenta.
- **Prompts largos:** Strata lee entero el primer mensaje de un chat, más o menos 1 minuto por cada 30,000 tokens. Los
  mensajes siguientes empiezan en segundos.

Más: [dónde se guardan tus chats](docs/INSTALL.md#where-things-are-stored), [la API](docs/DETAILS.md#using-it).

### Opcional: más velocidad con una sola tarjeta cuando todo el modelo cabe en la RAM

Dos ajustes opcionales (de RubenTorney, #1904 y #1905). Ayudan si usas una tarjeta gráfica y tu RAM guarda todos los expertos (el modo arena de Strata, por ejemplo Q2_0 o IQ3_XXS en un PC con 64 GB). Abre tu configuración `strata-<model>.json` en la carpeta de datos de Strata ([dónde](docs/INSTALL.md#where-things-are-stored)) y añade:

- **Generación más rápida:** `"--adapt-async", "1"` en la lista `"args"`. La caché de expertos se actualiza entonces en un hilo auxiliar en vez de pausar la respuesta. Medido en una RTX 5070 (Windows, Q2_0): +4-5 % de generación, +7-10 % en código y documentos largos. Las respuestas pueden diferir un poco de las por defecto (son iguales de una ejecución a otra).
- **Lectura del prompt más rápida en Windows:** `"STRATA_PIN_BY_NEED": "1"` en el bloque `"env"`. Windows solo puede bloquear cerca de la mitad de la RAM para la tarjeta; esto bloquea primero los expertos que la tarjeta necesita de verdad. Medido en la RTX 5070: +4,4 % en la lectura del prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Reinicia Strata (`START-HERE.bat` o `./setup.sh`) después del cambio. Para deshacerlo, quita las dos entradas.

## ¿Algo salió mal?

- **Mi PC se congeló la primera vez que arrancó Strata.** Es normal mientras carga el modelo. Espera y no cierres la
  ventana. ¿Sigue congelado después de 10 minutos? Reinicia el PC, cierra otros programas y vuelve a intentarlo, o
  elige un tamaño más pequeño.
- **Se detuvo durante la descarga o la instalación.** Vuelve a ejecutar `START-HERE.bat` (o `./setup.sh`). Sigue
  donde se detuvo.
- **Va muy lento y la luz del disco no deja de parpadear, o dice "the engine stopped unexpectedly".** Tu PC no tiene
  suficiente RAM libre. Cierra otros programas (los navegadores usan mucha), o elige un tamaño más pequeño (Q2_0 o
  IQ2_XS).
- **Dice que el puerto 8080 ya está en uso.** Strata ya está en marcha. Busca su ventana.

Más problemas y sus soluciones: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). ¿Sigues atascado? Abre un
[issue](https://github.com/Niko1221/Strata/issues) y adjunta `strata-<model>.log` de la carpeta de Strata. ¿Has
encontrado un problema de seguridad? Infórmalo en privado: [SECURITY.md](SECURITY.md).

## ¿Cómo funciona?

Los modelos como este suelen funcionar en servidores con cientos de gigabytes de memoria gráfica. Tu tarjeta gráfica
tiene 12-24 GB. Strata hace que el modelo quepa **repartiendo el trabajo por todo tu PC**. Piensa en una cocina: lo
que usas todo el rato se queda en la encimera, y el resto espera en la despensa.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="Los 24,576 expertos del modelo: los más usados en la tarjeta gráfica, todos en la RAM y una tabla de consulta en el SSD"></p>

- **El modelo es un equipo de 24,576 pequeños especialistas ("expertos").** Cada palabra solo necesita 10 de ellos.
- **Tu tarjeta gráfica** guarda los pocos miles de expertos que más se usan. **Tu RAM** los tiene todos, y **tu
  procesador** trabaja con el resto al mismo tiempo. **Tu SSD** guarda una gran tabla de consulta.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="Un pequeño ayudante adivina las siguientes palabras; el modelo grande las comprueba todas a la vez y se queda con las correctas"></p>

- **Adivinar y comprobar:** un pequeño ayudante adivina las siguientes palabras. El modelo grande las comprueba todas
  a la vez. Recibes la misma respuesta, 1.6-1.8x antes.
- **Los textos largos se leen en trozos grandes** (hasta 8,192 tokens cada vez), a más de 1,000 tokens por segundo.

La explicación larga: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Cada parte y sus números:
[los detalles](docs/DETAILS.md#how-it-works) y el [artículo](docs/paper/Strata-Paper.pdf).

## Créditos y licencia

El modelo es [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), del equipo de Qwen. Lo
comprimieron [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
y Unsloth. Strata usa partes de [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos los créditos:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata es de código abierto con la [licencia MIT](LICENSE).
Algunas partes y cada modelo tienen sus propias licencias ([cuáles](docs/HOW_IT_WORKS.md#license)).

## Apoya a Strata

Strata es libre y de código abierto. Si te resulta útil, puedes apoyar su desarrollo:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
