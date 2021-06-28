# Commenti di cumani (e miei)
Il report così com'è vale 15 punti
## Errors
* LDA non ha senso dato che fa diventare tutto compresso in una sola dimensione mentre lascia tutto il resto a 0 nel caso
di problemi binari, possibili soluzioni sono correggere il numero di features segnato a 1 e mantenere i risultati ottenuti,
oppure cambiare con o aggiungere la PCA che ti permette di mantenere quante features vuoi.
## Imperfections
* Gli hyperparameters usati per la SVM sono copiati da quelli del laboratorio e hanno poco senso nel nostro contesto,
studiati cosa significano e rifai l'hyperparameter tuning
* Utilizziamo solo la minDCF per fare model selection, cumani dice che avremmo dovuto utilizzare anche la actual DCF,
non so esattamente cosa significhi, controlla bene
  