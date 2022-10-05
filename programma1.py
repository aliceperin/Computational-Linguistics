import sys
import nltk

#trova numero delle frasi e dei token, con POS tag
def NFrasiToken(frasi):
   #inizializzo come liste vuote quelle dove inserirò le frasi totali e i token totali
   #e momentaneamente assegno 0 al numero totale di frasi e token
   listaFrasi=[]
   nFrasi=0
   listaToken=[]
   nToken=0
   #ciclo tutte le frasi del testo e vado ad aggiungerle 
   #ad una ad una alla lista delle frasi, contandole
   for frase in frasi:
      listaFrasi.append(frase)
      nFrasi+=1
   #... stessa cosa per i token. Li aggiungo alla lista e li conto usando un ciclo for
      tokens = nltk.word_tokenize(frase)
      for tok in tokens:
         listaToken.append(tok)
         nToken+=1
   #uso la funzione di nltk pos_tag per annotare con le part of speech
   #i token all'interno della lista dei token 
   tokenPOStag = nltk.pos_tag(listaToken)
   return nFrasi, nToken, listaToken, tokenPOStag

#trova lunghezza media delle frasi e delle parole
def LunghezzeMedie (nFrasi, nToken, listaToken):
#trovo la lunghezza media delle frasi con un rapporto tra i token 
#totali e il numero delle frasi trovati della funzione precedente
   lenMediaFrasi=nToken/nFrasi
#con lo stesso principio, inizializzo a 0 la lunghezza dei token in caratteri...
   lenTokens=0
#...ciclo ogni token all'interno nella lista, contando i suoi caratteri 
#e aggiungendoli alla variabile lenTokens
   for tok in listaToken:
      lenTokens+=len(tok)
#trovo la lunghezza media in caratteri con un rapporto tra 
#la totalità dei caratteri del testo e il numero di token
   lenMediaTok=lenTokens/nToken

   return lenMediaFrasi, lenMediaTok

#trova la totalità delle parole tipo e la TTR
def Vocab_TTR (listaToken, nToken):
#trovo il vocabolario indicizzando la lista token nel "range" da 0 a 5000 token
#"set" elimina i duplicati dei token, "list" trasforma il set in una lista
   vocabolario=list(set(listaToken[:5000]))
#trovo la lunghezza del vocabolario
   lenVocabolario=len(vocabolario)
#Indice di ricchezza lessicale dalla formula |Vocabolario|/|Corpus|
   TypeTokenRatio=lenVocabolario/nToken
   return lenVocabolario, TypeTokenRatio

#Trova la distribuzione delle classi di frequenza V1, V5, V10 
#all'aumentare del corpus di 500 token per volta
def ClassiDiFrequenza (listaToken, nToken):
#Creo un dizionario vuoto per ogni classe di frequenza in cui ci sarà come chiave la porzione di token
#e come valore la distribuzione di frequenza in quella forzione
   V1={}
   V5={}
   V10={}
#inizializzo un contatore per ciascuna classe di frequenza che dividerò 
#per il numero dei token totali per trovare la distribuzione della classe
#ogni 500 token del testo
   count1=0
   count5=0
   count10=0
#---Trovo la classe di frequenza V1 e per ogni porzione creo una nuova chiave nel rispettivo dizionario
#dove inserisco la distribuzione di frequenza facendo un rapporto
#tra i token che rientrano in quella classe e il numero totale dei token nel testo
   for tok in listaToken[:500]:
      if listaToken[:500].count(tok)==1:
         count1+=1
         V1["500"]=count1/nToken
   for tok in listaToken[500:1000]:
      if listaToken[500:1000].count(tok)==1:
         count1+=1
         V1["1000"]=count1/nToken
   for tok in listaToken[1000:1500]:
      if listaToken[1000:1500].count(tok)==1:
         count1+=1
         V1["1500"]=count1/nToken
   for tok in listaToken[1500:2000]:
      if listaToken[1500:2000].count(tok)==1:
         count1+=1
         V1["2000"]=count1/nToken
   for tok in listaToken[2000:2500]:
      if listaToken[2000:2500].count(tok)==1:
         count1+=1
         V1["2500"]=count1/nToken
   for tok in listaToken[2500:3000]:
      if listaToken[2500:3000].count(tok)==1:
         count1+=1
         V1["3000"]=count1/nToken
   for tok in listaToken[3000:3500]:
      if listaToken[3000:3500].count(tok)==1:
         count1+=1
         V1["3500"]=count1/nToken
   for tok in listaToken[3500:4000]:
      if listaToken[3500:4000].count(tok)==1:
         count1+=1
         V1["4000"]=count1/nToken
   for tok in listaToken[4000:4500]:
      if listaToken[4000:4500].count(tok)==1:
         count1+=1
         V1["4500"]=count1/nToken
   for tok in listaToken[4500:5000]:
      if listaToken[4500:5000].count(tok)==1:
         count1+=1
         V1["5000"]=count1/nToken

#---Trovo la classe di frequenza V5
   
   for tok in listaToken[:500]:
      if listaToken[:500].count(tok)==5:
         count5+=1
         V5["500"]=count5/nToken
   for tok in listaToken[500:1000]:
      if listaToken[500:1000].count(tok)==5:
         count5+=1
         V5["1000"]=count5/nToken
   for tok in listaToken[1000:1500]:
      if listaToken[1000:1500].count(tok)==5:
         count5+=1
         V5["1500"]=count5/nToken
   for tok in listaToken[1500:2000]:
      if listaToken[1500:2000].count(tok)==5:
         count5+=1
         V5["2000"]=count5/nToken
   for tok in listaToken[2000:2500]:
      if listaToken[2000:2500].count(tok)==5:
         count5+=1
         V5["2500"]=count5/nToken
   for tok in listaToken[2500:3000]:
      if listaToken[2500:3000].count(tok)==5:
         count5+=1
         V5["3000"]=count5/nToken
   for tok in listaToken[3000:3500]:
      if listaToken[3000:3500].count(tok)==5:
         count5+=1
         V5["3500"]=count5/nToken
   for tok in listaToken[3500:4000]:
      if listaToken[3500:4000].count(tok)==5:
         count5+=1
         V5["4000"]=count5/nToken
   for tok in listaToken[4000:4500]:
      if listaToken[4000:4500].count(tok)==5:
         count5+=1
         V5["4500"]=count5/nToken
   for tok in listaToken[4500:5000]:
      if listaToken[4500:5000].count(tok)==5:
         count5+=1
         V5["5000"]=count5/nToken

#trovo la classe di frequenza V10

   for tok in listaToken[:500]:
      if listaToken[:500].count(tok)==10:
         count10+=1
         V10["500"]=count10/nToken
   for tok in listaToken[500:1000]:
      if listaToken[500:1000].count(tok)==10:
         count10+=1
         V10["1000"]=count10/nToken
   for tok in listaToken[1000:1500]:
      if listaToken[1000:1500].count(tok)==10:
         count10+=1
         V10["1500"]=count10/nToken
   for tok in listaToken[1500:2000]:
      if listaToken[1500:2000].count(tok)==10:
         count10+=1
         V10["2000"]=count10/nToken
   for tok in listaToken[2000:2500]:
      if listaToken[2000:2500].count(tok)==10:
         count10+=1
         V10["2500"]=count10/nToken
   for tok in listaToken[2500:3000]:
      if listaToken[2500:3000].count(tok)==10:
         count10+=1
         V10["3000"]=count10/nToken
   for tok in listaToken[3000:3500]:
      if listaToken[3000:3500].count(tok)==10:
         count10+=1
         V10["3500"]=count10/nToken
   for tok in listaToken[3500:4000]:
      if listaToken[3500:4000].count(tok)==10:
         count10+=1
         V10["4000"]=count10/nToken
   for tok in listaToken[4000:4500]:
      if listaToken[4000:4500].count(tok)==10:
         count10=1
         V10["4500"]=count10/nToken
   for tok in listaToken[4500:5000]:
      if listaToken[4500:5000].count(tok)==10:
         count10+=1
         V10["5000"]=count10/nToken
     
   return V1, V5, V10

# trova la media dei sostantivi e verbi per frase 
def MediaSostantiviVerbi (tokenPOStag, nFrasi):
   listaSostantivi=[]
   listaVerbi=[]
#dal testo annotato con POS tag nella prima funzione, prendo il secondo 
# elemento del bigramma e se corrisponde a un sostantivo o a un verbo
#lo inserisco nella rispettiva lista
   for bigramma in tokenPOStag:
      if bigramma[1] in ["NN", "NNS", "NNP", "NNPS"]:
         listaSostantivi.append(bigramma[1])
      if bigramma[1] in ["VB", "VBD", "VBN", "VBP", "VBZ"]:
         listaVerbi.append(bigramma[1])
   
#trovo la quantità di tag nelle liste
   numeroSostantivi= len(listaSostantivi)
   numeroVerbi=len(listaVerbi)
#trovo la media con un rapporto tra il numero di tag corrispondenti a sostantivi e verbi 
#e il numero di frasi
   mediaSostantivi= numeroSostantivi/nFrasi
   mediaVerbi=numeroVerbi/nFrasi

   return mediaSostantivi, mediaVerbi

#trovo la densità lessicale, ovvero il rapporto tra quantità di sostantivi, verbi, avverbi e aggettivi
#e il numero totale di token nel corpus, non contando "." e ","
def DensitàLessicale (tokenPOStag):
   #conto la quantità di sostantivi, verbi, aggettivi, avverbi
   listaSVAA=[]
   for bigramma in tokenPOStag:
      if bigramma[1] in ["VB", "VBD", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"]:
         listaSVAA.append(bigramma[1])
   quantitàSVAA=len(listaSVAA)

   #conto la totalità dei token eliminando punti e virgole 
   listaPOS_noPV=[]
   listaPV=[]
   for bigramma in tokenPOStag:
      if bigramma[1] in [".", ","]:
         listaPV.append(bigramma[1])
      else:
         listaPOS_noPV.append(bigramma[1])
   tokenPOS_noPV= len(listaPOS_noPV)
   densitàLessicale = quantitàSVAA/tokenPOS_noPV

   return densitàLessicale

def main (file1, file2):
   fileInput1=open(file1, mode="r", encoding="utf-8")
   fileInput2=open(file2, mode="r", encoding="utf-8")
   raw1=fileInput1.read()
   raw2=fileInput2.read()
   sent_tokenizer=nltk.data.load("tokenizers/punkt/english.pickle")
   frasi1=sent_tokenizer.tokenize(raw1)
   frasi2=sent_tokenizer.tokenize(raw2)

   # --- Chiamata delle funzioni e stampa risultati --- 

   # a) confronto il numero delle frasi e dei token dei due file
   print()
   nFrasiBiden, nTokenBiden, listaTokenBiden, POStagBiden=NFrasiToken(frasi1)
   print("Numero frasi Biden:", nFrasiBiden, "\tNumero token Biden:", nTokenBiden)
   nFrasiTrump, nTokenTrump, listaTokenTrump, POStagTrump=NFrasiToken(frasi2)
   print("Numero frasi Trump:", nFrasiTrump, "\tNumero token Trump:", nTokenTrump)
   print()
   print("****************")
   # b) confronto la lunghezza media delle frasi e dei token
   LunghezzaMediaFrasiB, LunghezzaMediaParoleB=LunghezzeMedie(nFrasiBiden, nTokenBiden, listaTokenBiden)
   print("La lunghezza media delle frasi di Biden è di: ", LunghezzaMediaFrasiB, " parole.\t", "La lunghezza media delle parole di Biden è di: ", LunghezzaMediaParoleB, "caratteri.")
   LunghezzaMediaFrasiT, LunghezzaMediaParoleT=LunghezzeMedie(nFrasiTrump, nTokenTrump, listaTokenTrump)
   print("La lunghezza media delle frasi di Trump è di: ", LunghezzaMediaFrasiT, " parole.\t", "La lunghezza media delle parole di Trump è di: ", LunghezzaMediaParoleT, "caratteri.")
   print()
   print("****************")
   # c) confronto le lunghezze dei vocabolari e le ricchezze lessicali (Type Token Ratio)
   lenVocabolarioBiden, TTRBiden=Vocab_TTR(listaTokenBiden, nTokenBiden)
   print("Vocabolario Biden: ", lenVocabolarioBiden, "parole tipo.", "\tRicchezza lessicale: ", TTRBiden)
   lenVocabolarioTrump, TTRTrump=Vocab_TTR(listaTokenTrump, nTokenTrump)
   print("Vocabolario Trump: ", lenVocabolarioTrump, "parole tipo.", "\tRicchezza lessicale: ", TTRTrump)
   print()
   print("****************")
   # d) confronta la distribuzione delle classi di frequenza V1, V5 e V10 per porzioni incrementali di 500 token
   classeV1B, classeV5B, classeV10B= ClassiDiFrequenza(listaTokenBiden, nTokenBiden)
   print("Distribuzione classi di frequenza Biden ogni 500 token:")
   print()
   print("Classe di frequenza V1:", classeV1B)
   print()
   print("Classe di frequenza V5:", classeV5B)
   print()
   print("Classe di frequenza V10:", classeV10B)
   print()
   print("****************")
   classeV1T, classeV5T, classeV10T= ClassiDiFrequenza(listaTokenTrump, nTokenTrump)
   print("Distribuzione classi di frequenza Trump ogni 500 token:")
   print()
   print("Classe di frequenza V1:", classeV1T)
   print()
   print("Classe di frequenza V5:", classeV5T)
   print()
   print("Classe di frequenza V10:", classeV10T)
   print()
   print("****************")
   # e) stampa la media dei sostantivi e quella dei verbi per frase
   mediaSostantiviBiden, mediaVerbiBiden = MediaSostantiviVerbi(POStagBiden, nFrasiBiden)
   print("Biden, in media per frase ha: ", mediaSostantiviBiden, " sostantivi\n\t\t\t\t", mediaVerbiBiden, " verbi")
   print()

   mediaSostantiviTrump, mediaVerbiTrump = MediaSostantiviVerbi(POStagTrump, nFrasiTrump)
   print("Trump, in media per frase ha: ", mediaSostantiviTrump, " sostantivi\n\t\t\t\t", mediaVerbiTrump, " verbi")
   print()
   print("****************")
   # f) stampa la densità lessicale
   densitàLessicaleBiden= DensitàLessicale(POStagBiden)
   print("La densità lessicale di Biden è: ", densitàLessicaleBiden)
   print()
   densitàLessicaleTrump= DensitàLessicale(POStagTrump)
   print("La densità lessicale di Trump è: ", densitàLessicaleTrump)
   print()
   print("****************")
main(sys.argv[1], sys.argv[2])