Avvio applicazione

## Avvio applicazione

```bash
dotnet run
```

Per risolvere il problema `Unhandled exception. System.IO.IOException: The configured user limit (128) on the number of inotify instances has been reached ...` eseguire i seguenti passi:

 - sudo nano /etc/sysctl.conf
 - add those lines to bottom of file:
   ```
   fs.inotify.max_user_watches = 1638400
   fs.inotify.max_user_instances = 1638400
   ```
 - save the file
 - sudo sysctl -p


Serve per caricare davvero nel Progetto I package appena scaricati
```bash
dotnet restore
```

## Generazione codice

Generazione di un controller:

```bash
dotnet-aspnet-codegenerator controller -name StagioneController -async -api -m Stagione -dc StagioneContext -outDir Controllers
```

Per far funzionare il generatore di codice su linux non funzionano le ultime versioni di dotnet-sdk e dotnet-aspnet-codegenerator, bisogna fare il downgrade di una versione ad entrambi. Vedi qui: https://github.com/dotnet/Scaffolding/issues/1384#issuecomment-689293100


Per risolvere il bug ho anche dovuto fare il downgrade di CodeGeneration.Design dalla v.3.1.4 alla v.3.1.3
```bash
# downgrade del pacchetto dotnet-sdk-3.1 dalla v.3.1.401-1 alla v.3.1.302-1
dotnet tool install --global dotnet-aspnet-codegenerator --version 3.1.3
dotnet add package Microsoft.VisualStudio.Web.CodeGeneration.Design --version 3.1.3
```
