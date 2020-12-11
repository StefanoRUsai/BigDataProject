Per far partire il progetto sulle macchine AWS, come scritto precedentemente, abbiamo utilizzato \emph{Terraform}. Qui di seguito elenchiamo i comandi per far partire il progetto.
Per far partire il progetto bisogna aver installato nel proprio ambiente di lavoro aws-cli, essere in possesso di un account AWS e essere in grado di utilizzare il protocollo SSH.
 Posizionarsi sulla cartella Terraform-Spark
 Digitare: \texttt{terraform init}
 Digitare: \texttt{terraform apply}
 Connettersi alle istanze tramite il link che si trova sulla piattaforma AWS
 Impostare nel file etc/hosts di sistema gli ip di ogni istanza
 Digitare: 
    hdfs namenode -format
    $HADOOP\_HOME/sbin/start-dfs.sh
    $HADOOP\_HOME/sbin/start-yarn.sh
    $HADOOP\_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
    ./spark/sbin/start-all.sh per far partire sull'istanza principale Spark e un worker
Digitare sulle istanze secondarie: \texttt{./spark/sbin/start-slave.sh A} (dove l'address è l'indirizzo ip fornito da spark)
Digitare: ./spark/bin/spark-submit --executor-memory 7g --driver-memory 120g  --num-executors 5 --executor-cores 7\\  --master spark://ec2-18-212-141-156.compute-1.amazonaws.com:7077 sentiment_analysis_mlspark.py


Istruzioni per far partire la webApp su un'ulteriore istanza aws:
All'interno dei file del frontend, nella cartella src, occorre modificare gli IP in base a quello pubblico della macchina.
Installare anche su questa macchina aws-cli, o creare una cartella
.aws con i file \emph{config} e \emph{credentials} in modo da poter accedere ad AWS. 


Posizionarsi sulla cartella Terraform-Node
Aprire 3 tre terminali
Inserire il token dei propri permessi AWS per poter usufruire del dataset su Athena AWS
Digitare nel terminale 1:
cd backendFlask
    python3 main.py
Digitare nel terminale 2:
   cd backend 
    npm start
Digitare nel terminale 3:
   cd frontEnd
    npm start

In alternativa la webAPP può essere utilizzata su un computer locale che abbia nel proprio ambiente aws-cli.
