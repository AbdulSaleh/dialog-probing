

# Pruebas a Sistemas de Diálogo Neural para la Comprensión Conversacional
Este código acompaña al artículo "[Probing Neural Dialog Models for Conversational Understanding](https://arxiv.org/abs/2006.08331)" de Saleh et al., 2020. 

Este repositorio se construye sobre [ParlAI](https://parl.ai/). Añadimos funcionalidad para probar modelos de diálogo de dominio abierto (RNN y Transformers). 
Las pruebas evalúan la calidad de las representaciones internas del modelo para habilidades conversacionales. 

## Configuración

Siga las mismas instrucciones de instalación que [ParlAI](https://github.com/facebookresearch/ParlAI/tree/d510bc2e10633d5204e1957a6c98cf30aa1be10d). ParlAI requiere Python 3 y PyTorch 1.1 o superior. Después de clonar este repositorio, recuerde ejecutar 

```
python setup.py develop
```

También deberá instalar [skorch](https://github.com/skorch-dev/skorch/tree/14f374db158ec7a7f4770a2fa9b02b8016d2d6ff) 0.6, que es requerido por el clasificador de pruebas.  

## Ejemplo de Uso

Esta sección le guía a través de un ejemplo de cómo entrenar y probar un modelo de diálogo. 

1. Primero necesitará un modelo para probar. Entrenemos una RNN pequeña en el conjunto de datos DailyDialog:

    ```
    python examples/train_model.py -t dailydialog -m seq2seq --bidirectional true --numlayers 2 --hiddensize 256 --embeddingsize 128 -eps 60 -veps 1 -vp 10 -bs 32 --optimizer adam --lr-scheduler invsqrt -lr 0.005 --dropout 0.3 --warmup-updates 4000 -tr 300  -mf trained/dailydialog/seq2seq/seq2seq --display-examples True -ltim 30 --tensorboard_log True --validation-metric ppl
    ```

    Para entrenar con datos perturbados (es decir, aleatorizados), agregue la bandera ``-sh within``. Consulte la [documentación](http://parl.ai.s3-website.us-east-2.amazonaws.com/docs/index.html) de ParlAI para obtener más información sobre el entrenamiento de modelos de diálogo. 

2. Luego, generará y guardará las representaciones vectoriales para usarlas como características en el clasificador de pruebas. 
Generemos y guardemos los vectores ``encoder_state`` para la tarea de clasificación de preguntas TREC:

    ```
    python probing/probe_model.py -mf trained/dailydialog/seq2seq/seq2seq -t probing.tasks.trecquestion.agents --probe encoder_state 
    ```
    Esto descargará automáticamente los datos de la tarea requeridos y guardará las representaciones generadas en ``trained/dailydialog/seq2seq/probing/encoder_state/trecquestion``.
    
    Existen tres tipos de representaciones ocultas que puede extraer y probar:
    * `encoder_state`
    * `word_embeddings`
    * `combined`
    
    Consulte el [artículo](https://arxiv.org/abs/2006.08331) para obtener más información. 

3. Ahora puede ejecutar el clasificador de pruebas para evaluar la calidad de las representaciones generadas mediante:

    ```
    python probing/eval_probing.py -m trained/dailydialog/seq2seq -t trecquestion --probing-module encoder_state --max_epochs 50 --runs 30
    ```
    Esto entrena el clasificador de pruebas (una MLP) con las representaciones generadas. Los resultados finales se guardan en ``trained/dailydialog/seq2seq/probing/encoder_state/trecquestion/results.json``. 

### Líneas Base

1. También puede que desee generar las líneas base de incrustaciones de palabras GloVe. Puede hacerlo ejecutando:

    ```
    python probing/glove.py -t trecquestion --dict-path trained\dailydialog\seq2seq\seq2seq.dict
    ```
    Esto descargará automáticamente las incrustaciones GloVe y guardará las representaciones generadas en ``trained/GloVe/probing/trecquestion``

2. Ahora debe ejecutar el clasificador de pruebas en estas representaciones generadas utilizando:

    ```
    python probing/eval_probing.py -m GloVe -t trecquestion --max_epochs 50 --runs 30
    ```
    Los resultados finales se guardan en ``trained/GloVe/probing/trecquestion/results.json``. 

**Nota importante:** Se recomienda encarecidamente utilizar los mismos directorios que en los ejemplos para guardar sus puntos de control y datos, de lo contrario, otras partes del código se romperán.

## Tareas de pruebas
Las tareas de pruebas soportadas son:

* trecquestion 
* multiwoz
* sgd
* dialoguenli
* wnli
* snips
* scenariosa
* dailydialog_topic

### Agregar nuevas tareas de pruebas
Las nuevas tareas de pruebas deben seguir el siguiente formato:
```
text: <utterance1> \n
<utterance2> \t episode_done:True \n  
text: <utterance1> \n
<utterance2> \n
<utterance3> \t episode_done:True \n 
... 
```
Las nuevas tareas de pruebas deben agregarse a [**probing/tasks**](./probing/tasks) 
y a [glove.py](./probing/glove.py).

El código es más adecuado para tareas donde la etiqueta se basa en:
* todos los enunciados en un diálogo (como DailyDialog Topic)
* o la interacción entre dos enunciados (como DialogueNLI)
* o el último enunciado en un diálogo (como ScenarioSA)

Consulte la sección [3.2](https://arxiv.org/abs/2006.08331) del artículo para más información. 


## Organización del Código

La mayor parte del código para este estudio se encuentra dentro del directorio [**probing**](./probing). 

También ampliamos [torch_generator_agent.py](./parlai/core/torch_generator_agent.py) con funciones de pruebas que extraen las representaciones internas de un modelo. 


## Referencia

Si utiliza este código, por favor cite nuestro [artículo](https://arxiv.org/abs/2006.08331):
```
@article{saleh2020probing,
    author= {Saleh, Abdelrhman and Deutsch, Tovly and Casper, Stephen and Belinkov, Yonatan and Shieber, Stuart},
    title= {Probing Neural Dialog Models for Conversational Understanding},
    journal= {Second Workshop on NLP for Conversational AI},
    year= {2020}
    }
```
