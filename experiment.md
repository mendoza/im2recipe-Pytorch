# Primera compilación

01. Compilar Pytorch-wordemb como fue sugerido en el repo.
02. Descargar `vocab.bin.gz`.
03. Descomprimir todo a `./data`
04. Correr `python get_vocab.py ../data/vocab.bin` dentro de `./scripts`
05. Descargar `recipe1M_layers.tar.gz`.
06. Descomprimir todo a `./data/recipe1M`
07. Descargar `det_ingrs.json` a `./data/recipe1M/det_ingrs.json`
08. Descargar los conjutos de datos de NLTK punkt y stopwords
09. Correr `python bigrams.py --crtbgrs`
10. Correr `python bigrams.py --nocrtbgrs`
11. Esto crea `classes1M.pkl`
12. Correr `python tokenize_instructions.py train` para crear el archivo para entrenar word2vec
13. Correr `python tokenize_instructions.py` para la partision de datos para skip-thought (más adelante)
14. Correr `./word2vec -hs 1 -negative 0 -window 10 -cbow 0 -iter 10 -size 300 -binary 1 -min-count 10 -threads 20 -train ../../data/tokenized_instructions_train.txt -output vocab.bin`
15. Correr `python get_vocab.py ../packages/word2vec/vocab.bin`
16. Pasar el vocab.bin y el vocab.txt (generado por el paso 15) a `data/text`
17. Descargar encs_train_1024.t7, encs_val_1024.t7 y encs_test_1024.t7 (skip-thoughts)
18. Correr `python mk_dataset.py --vocab ../data/text/vocab.txt`
19. Esto generara el dataset con una modificación solo tenda el `15%` de los datos disponibles
20. Eran originalmente `1,029,720 elementos` distribuidos en: entrenamiento, `254247 elementos`; validación, `54565 elementos`; test, `54887 elementos`.
21. Correr `crop_dataset.py` para reducir los elementos por un porcentaje `--percentage 0.85`

# Cosas interesantes
1. Number of unique ingredients 17232
2. Number of unique ingredients 18253
3. Se utilizaran maximo 5 imagenes por receta (Revisar más adelante)