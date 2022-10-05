import sys
import math
import nltk
from nltk import bigrams

# definisco una funzione che annota il testo con i POS tag
def Annotazione (frasi):
   tokensTOT=[]
   tokensPOStot=[]
   for frase in frasi:
      tokens=nltk.word_tokenize(frase)
      tokensPOS=nltk.pos_tag(tokens)
      tokensPOStot+=tokensPOS
      tokensTOT+=tokens
   lunghezzaCorpus=len(tokensTOT)
   bigrammiDiToken=list(nltk.bigrams(tokensTOT))
   fdist=nltk.FreqDist(tokensTOT)
   return tokensTOT, tokensPOStot, lunghezzaCorpus, bigrammiDiToken, fdist

#definisco una funzione che trova i primi 10 POS tag più frequenti
#e i primi 20 verbi più frequenti e i primi 20 sostantivi più frequenti
def POSfrequenti (tokensPOStot):
#inizializzo un dizionario vuoto per contare la quantità di ciascun tag...
   dictTag = {}
#...in cui inserisco una nuova chiave se non esiste (assegnandole il valore 1)
#mentre se già esiste incremento la sua quantità (valore) di 1 
   for bigramma in tokensPOStot:
      if bigramma[1] not in dictTag:
         dictTag[bigramma[1]]= 1
      else:
         dictTag[bigramma[1]] += 1
   
#ordino il dizionario per frequenza decrescente
   dictTagOrdinato= sorted(dictTag.items(), key=lambda x: x[1], reverse=True)

#trovo i 20 sostantivi e i 20 verbi più utilizzati

#creo due liste vuote in modo da dividere i tag e i relativi token nella lista di riferimento
   listaSostantivi=[]
   listaVerbi=[]
   for bigramma in tokensPOStot:
      if bigramma[1] in ["NN","NNS", "NNP", "NNPS"]:
         listaSostantivi.append(bigramma)
      if bigramma[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
         listaVerbi.append(bigramma)

#creo dei dizionari vuoti
   sostantivi={}
   verbi={}
   for bigramma in listaSostantivi:
#se il valore all'indice 0 del bigramma ovvero il token, non è presente nel dizionario,
#viene creato come chiave con valore 1.
      if bigramma[0] not in sostantivi:
         sostantivi[bigramma[0]]=1
#se invece esiste già, il suo valore (frequenza) viene incrementato di 1
      else:
         sostantivi[bigramma[0]]+=1
#la stessa procedura vale anche per i verbi
   for bigramma in listaVerbi:
      if bigramma[0] not in verbi:
         verbi[bigramma[0]]=1
      else:
         verbi[bigramma[0]]+=1
#ordino gli elementi all'interno dei dizionari in ordine decrescente di frequenza
   sostantivi_20=sorted(sostantivi.items(), key=lambda x: x[1], reverse=True)
   verbi_20=sorted(verbi.items(), key=lambda x: x[1], reverse=True)

   return dictTagOrdinato, sostantivi_20, verbi_20
#estraggo i bigrammi sostantivo-verbo e aggettivo-sostantivo più frequenti

def EstraiCoppieBigrammi(tokensPOStot):
#divido in bigrammi il testo annotato
   bigrammaTokPOS_SV=bigrams(tokensPOStot)
#scorro la lista dei bigrammi e inserisco nella lista vuota i bigrammi che mi interessano,
#quindi le coppie di bigrammi annotate sostantivo-verbo
   SVbigrammiEstratti=[]
   for bigramma in bigrammaTokPOS_SV:
      if ((bigramma[0][1] in ["NNP", "NNPS", "NN", "NNS"]) and (bigramma[1][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])):
         SVbigrammiEstratti.append(bigramma)

#inizializzo un dizionario in cui andrò a mettere i bigrammi sostantivo-verbo più frequenti
   SostantiviVerbi={}
   TypeBigrammiSV=set(SVbigrammiEstratti)
   for bigramma in TypeBigrammiSV:
      freq = SVbigrammiEstratti.count(bigramma)
      SostantiviVerbi[bigramma]=freq

   bigrammaTokPOS_AS=bigrams(tokensPOStot)
#scorro la lista dei bigrammi e inserisco nella lista vuota i bigrammi che mi interessano,
#quindi le coppie di bigrammi annotate aggettivo-sostantivo
   ASbigrammiEstratti=[]
   for bigramma in bigrammaTokPOS_AS:
      if ((bigramma[0][1] in ["JJ", "JJR", "JJS"]) and (bigramma[1][1] in ["NN", "NNS", "NNP", "NNPS"])):
         ASbigrammiEstratti.append(bigramma)

#inizializzo un dizionario in cui andrò a mettere i bigrammi aggettivo-sostantivo più frequenti
   AggettiviSostantivi={}
   TypeBigrammiAS=set(ASbigrammiEstratti)
   for bigramma in TypeBigrammiAS:
      freq = ASbigrammiEstratti.count(bigramma)
      AggettiviSostantivi[bigramma]=freq
   
   sostantiviVerbi_ordinato=sorted(SostantiviVerbi.items(), key=lambda x: x[1], reverse=True)
   aggettiviSostantivi_ordinato=sorted(AggettiviSostantivi.items(), key=lambda x: x[1], reverse=True)

   return sostantiviVerbi_ordinato, aggettiviSostantivi_ordinato

def Probabilità_LMI (lunghezzaCorpus, fdist, bigrammiDiToken):
#estraggo i bigrammi che hanno frequenza>3 e li inserisco in una lista vuota
   bigrammiFreq3=[]
   for bigramma in bigrammiDiToken:
      if bigrammiDiToken.count(bigramma)>3:
         bigrammiFreq3.append(bigramma)

#creo un dizionario vuoto per ogni informazione da calcolare
   Pcongiunta={}
   Pcondizionata={}
   Local_MI={}
#con un ciclo for, per ogni bigramma...
   for bigramma in bigrammiFreq3:
#...calcolo la frequenza nel testo di entrambi i token da cui è formato, estraendo l'informazione 
#...dalla distribuzione di frequenza
      FreqToken1= fdist[bigramma[0]]
      FreqToken2= fdist[bigramma[1]]
#...calcolo la frequenza del bigramma
      freqBigramma= bigrammiDiToken.count(bigramma)

#Sempre all'interno del ciclo for, calcolo la probabilità congiunta, condizionata e la LMI con le relative formule
      probabilitàCongiunta=(freqBigramma/lunghezzaCorpus)
#inserisco nel dizionario come chiave il bigramma e come valore la sua probabilità congiunta
      Pcongiunta[bigramma]=probabilitàCongiunta
#inserisco nel dizionario come chiave il bigramma e come valore la sua probabilità condizionata
      probabilitàCondizionata=(FreqToken1/lunghezzaCorpus)*(freqBigramma/FreqToken2)
      Pcondizionata[bigramma]=probabilitàCondizionata
#inserisco nel dizionario come chiave il bigramma e come valore la sua LMI
      LocalMI=freqBigramma*(math.log2((freqBigramma*lunghezzaCorpus)/(FreqToken1*FreqToken2)))
      Local_MI[bigramma]=LocalMI

   return Pcongiunta, Pcondizionata, Local_MI

#funzione per ordinare i valori di un dizionario in ordine decrescente
def OrdinaDizionario (dizionario):
   return sorted(dizionario.items(), key=lambda x: x[1], reverse=True)

#funzione che smista le frasi in base alla loro lunghezza, da lunghezza  8 token a lunghezza 15 token,
#ne calcola la probabilità con un modello di Markov di ordine 1, con l'Add-One Smoothing,
#ordina i dizionari contenente le frasi  per probabilità decrescente 
#stampa la frase con probabilità maggiore, e la relativa probabilità 
def SmistaLunghezzeFrasi_Markov (frasi, lunghezzaCorpus, tokensTOT):
#creo dizionari vuoti, una per ogni lunghezza di frase
   frasiDa8={}
   frasiDa9={}
   frasiDa10={}
   frasiDa11={}
   frasiDa12={}
   frasiDa13={}
   frasiDa14={}
   frasiDa15={}
#trovo la distribuzione di frequenza dei token
   fdistTokens=nltk.FreqDist(tokensTOT)
#trovo la quantità di parole tipo
   vocabolario=len(set(tokensTOT))
#creo una lista di tutti i bigrammi del testo
   bigrammiTesto=list(nltk.bigrams(tokensTOT))
#...e trovo la loro distribuzione di frequenza
   fdistBigrammi=nltk.FreqDist(bigrammiTesto)

#scrivo un ciclo for per smistare le frasi nel dizionario di riferimento in base alla loro lunghezza:
   for frase in frasi:
#per trovare la lunghezza, devo prima tokenizzare la frase
      tokensFrase=nltk.word_tokenize(frase)
#e contare i token di cui è composta
      lenFrase=len(tokensFrase)
#poi in base al numero di token, inserisco la frase come chiave del rispettivo dizionario.
#assegno come valore 1.0 di default a ciascuna frase.
      if lenFrase==8:
         frasiDa8[frase]=1.0
      if lenFrase==9:
         frasiDa9[frase]=1.0
      if lenFrase==10:
         frasiDa10[frase]=1.0
      if lenFrase==11:
         frasiDa11[frase]=1.0
      if lenFrase==12:
         frasiDa12[frase]=1.0
      if lenFrase==13:
         frasiDa13[frase]=1.0
      if lenFrase==14:
         frasiDa14[frase]=1.0
      if lenFrase==15:
         frasiDa15[frase]=1.0

# creo un ciclo for per ogni frase del dizionario, in cui:
      for frase in frasiDa8:
#tokenizzo la frase e ne estraggo una lista di bigrammi
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
#ne identifico il primo token e ne  calcolo la probabilità con un modello di markov del primo ordine e l'add-one smoothing
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
#inizializzo le variabili delle probabilità
         prob_frase=1.0
         prob_bigrammi=1.0
#creo un ciclo for in cui per ogni bigramma della frase:
         for bigramma in bigrammiFrase:
#trovo la sua frequenza 
            freqBigramma=fdistBigrammi[bigramma]
#trovo la frequenza del primo token del bigramma
            freqToken0bigramma=fdistTokens[bigramma[0]]
#trovo la probabilità del bigramma con il modello di Markov di ordine 1 con add-one smoothing
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
#moltiplico questa probabilità inizialmente a 1.0 (valore della variabile prob_bigrammi)
#che accumulerà i valori delle probabilità di ogni bigramma, ogni volta che ne moltiplicherà una per il valore precedente
            prob_bigrammi*=prob_bigramma
#moltiplica la probabilità del primo token per la probabilità dei bigrammi
         prob_frase= prob_primoToken * prob_bigrammi
#infine, assegna la probabilità della frase come valore della rispettiva chiave (frase)
         frasiDa8[frase]=prob_frase
         

      for frase in frasiDa9:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa9[frase]=prob_frase

      for frase in frasiDa10:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa10[frase]=prob_frase

      for frase in frasiDa11:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa11[frase]=prob_frase

      for frase in frasiDa12:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa12[frase]=prob_frase

      for frase in frasiDa13:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa13[frase]=prob_frase

      for frase in frasiDa14:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa14[frase]=prob_frase

      for frase in frasiDa15:
         fraseTokenizzata=list(nltk.word_tokenize(frase))
         bigrammiFrase=list(nltk.bigrams(fraseTokenizzata))
         primoToken=fraseTokenizzata[0]
         prob_primoToken=(fdistTokens[primoToken]+1)/(lunghezzaCorpus+vocabolario)
         prob_frase=1.0
         prob_bigrammi=1.0

         for bigramma in bigrammiFrase:
            freqBigramma=fdistBigrammi[bigramma]
            freqToken0bigramma=fdistTokens[bigramma[0]]
            prob_bigramma=(freqBigramma+1)/(freqToken0bigramma+vocabolario)
            prob_bigrammi*=prob_bigramma
         prob_frase= prob_primoToken * prob_bigrammi
         frasiDa15[frase]=prob_frase

   return frasiDa8, frasiDa9, frasiDa10, frasiDa11, frasiDa12, frasiDa13, frasiDa14, frasiDa15

#funzione che a partire dal POS tag, trova le NE (Entità Nominate)
def NamedEntity_tag (tokenPOStot):
   EntitaNominateTesto=nltk.ne_chunk(tokenPOStot)
#creo un dizionario per ogni entità nominata:
   person={}
   geoPE={}
   org={}

#classifico le NE nei rispettivi dizionari e assegno il valore 1 se il nodo non è presente nel dizionario
# se è presente invece incremento di 1 il suo valore
   for nodo in EntitaNominateTesto:
      if hasattr(nodo, "label"):
         
         if nodo.label() in ["PERSON"]:
            for partNE in nodo.leaves():
               if partNE not in person:
                  person[partNE]=1
               else: 
                  person[partNE]+=1

         if nodo.label() in ["GPE"]:
            for partNE in nodo.leaves():
               if partNE not in geoPE:
                  geoPE[partNE]=1
               else: 
                  geoPE[partNE]+=1

         if nodo.label() in ["ORGANIZATION"]:
           for partNE in nodo.leaves():
               if partNE not in org:
                  org[partNE]=1
               else: 
                  org[partNE]+=1
         

   return person, geoPE, org

def main (file1, file2):
#Biden.txt
   fileInput1=open(file1, mode="r", encoding="utf-8")
#Trump.txt
   fileInput2=open(file2, mode="r", encoding="utf-8")
   raw1=fileInput1.read()
   raw2=fileInput2.read()
   sent_tokenizer=nltk.data.load("tokenizers/punkt/english.pickle")
   frasi1=sent_tokenizer.tokenize(raw1)
   frasi2=sent_tokenizer.tokenize(raw2)

#--- Chiamo le funzioni e stampo i risultati ---
   tokensTOTbiden, tokensPOStotBiden, lunghezzaCorpusBiden, bigrammiTestoBiden, DistribuzioneFreqBiden = Annotazione (frasi1)
   tokensTOTtrump, tokensPOStotTrump, lunghezzaCorpusTrump, bigrammiTestoTrump, DistribuzioneFreqTrump = Annotazione (frasi2)

# a.1) stampo i primi 10 POS tag più frequenti nei rispettivi corpora
   dictTagBiden, sostantivi_20Biden, verbi_20Biden = POSfrequenti(tokensPOStotBiden)
   print("10 POS tag più frequenti per Biden:")
   print()
   for item in dictTagBiden[:10]:
      print (item)
   print()
   dictTagTrump,sostantivi_20Trump, verbi_20Trump = POSfrequenti(tokensPOStotTrump)
   print("10 POS tag più frequenti per Trump:")
   print()
   for item in dictTagTrump[:10]:
      print (item)
   print()
   print("************")
# a.2) stampo i primi 20 sostantivi e i primi 20 verbi per frequenza 
   print()
   print("I primi 20 sostantivi più frequenti di Biden:")
   print()
   for item in sostantivi_20Biden[:20]:
      print (item)
   print()
   print("I primi 20 sostantivi più frequenti di Trump:")
   print()
   for item in sostantivi_20Trump[:20]:
      print (item)
   print()
   print("************")
   print()
   print("I primi 20 verbi più frequenti di Biden:")
   print()
   for item in verbi_20Biden[:20]:
      print (item)
   print()
   print("I primi 20 verbi più frequenti di Trump:")
   print()
   for item in verbi_20Trump[:20]:
      print (item)
   print()
   print("************")

# a.3) stampo i 20 bigrammi sostantivo-verbo e aggettivo-sostantivo più frequenti
   bigrammiSV_Biden, bigrammiAS_Biden= EstraiCoppieBigrammi(tokensPOStotBiden)
   bigrammiSV_Trump, bigrammiAS_Trump=EstraiCoppieBigrammi(tokensPOStotTrump)
   
   print()
   print("I 20 bigrammi sostantivo-verbo più frequenti di Biden:")
   print()
   for item in bigrammiSV_Biden[:20]:
      print(item)
   print()
   print("I 20 bigrammi sostantivo-verbo più frequenti di Trump:")
   print()
   for item in bigrammiSV_Trump[:20]:
      print(item)
   print()
   print("************")
   print()
   print("I 20 bigrammi aggettivo-sostantivo più frequenti di Biden:")
   print()
   for item in bigrammiAS_Biden[:20]:
      print(item)
   print()
   print("I 20 bigrammi aggettivo-sostantivo più frequenti di Trump:")
   print()
   for item in bigrammiAS_Trump[:20]:
      print(item)
   print()
   print("************")

#b) stampo i 20 bigrammi token (frequenza minima: 3) con massima probabilità congiunta,
#  massima probabilità condizionata e massima LMI con relative probabilità e forza associativa
   PcongiunteBiden, PcondizionateBiden, LMIBiden =Probabilità_LMI(lunghezzaCorpusBiden, DistribuzioneFreqBiden, bigrammiTestoBiden)
   PcongiunteBidenOrdinato=OrdinaDizionario(PcongiunteBiden)
   PcondizionateBidenOrdinato=OrdinaDizionario(PcondizionateBiden)
   LMIBidenOrdinato=OrdinaDizionario(LMIBiden)
   print("Biden - 20 bigrammi con probabilità congiunta più alta:")
   print("Bigramma:\tProbabilità congiunta:")
   
   contatore = 0
   for key, value in PcongiunteBidenOrdinato:
      if contatore == 20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore+=1
         print(key, value)
   print()
   
   
   print("Biden - 20 bigrammi con probabilità condizionata più alta:")
   print("Bigramma:\tProbabilità condizionata:")
   contatore1=0
   for key, value in PcondizionateBidenOrdinato:
      if contatore1==20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore1+=1
         print(key, value)
   print()

   print("Biden - 20 bigrammi con LMI più alta:")
   print("Bigramma:\tLMI:")
   contatore2=0
   for key, value in LMIBidenOrdinato:
      if contatore2==20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore2+=1
         print(key, value)
   print()
   
   PcongiunteTrump, PcondizionateTrump, LMITrump =Probabilità_LMI(lunghezzaCorpusTrump, DistribuzioneFreqTrump, bigrammiTestoTrump)
   PcongiunteTrumpOrdinato=OrdinaDizionario(PcongiunteTrump)
   PcondizionateTrumpOrdinato=OrdinaDizionario(PcondizionateTrump)
   LMITrumpOrdinato=OrdinaDizionario(LMITrump)
   print("Trump - 20 bigrammi con probabilità congiunta più alta:")
   print("Bigramma:\tProbabilità congiunta:")
   contatore3=0
   for key, value in PcongiunteTrumpOrdinato:
      if contatore3==20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore3+=1
         print(key, value)
   print()
   
   print("Trump - 20 bigrammi con probabilità condizionata più alta:")
   print("Bigramma:\tProbabilità condizionata:")
   contatore4=0
   for key, value in PcondizionateTrumpOrdinato:
      if contatore4 ==20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore4+=1
         print(key, value)
   print()

   print("Trump - 20 bigrammi con LMI più alta:")
   print("Bigramma:\tLMI:")
   contatore5=0
   for key, value in LMITrumpOrdinato:
      if contatore5==20:
         break
      if (key[0] or key[1]) not in [".", ","]:
         contatore5+=1
         print(key, value)
   print()
   print("************")

#c) Stampo per ogni lunghezza di frase (da 8 a 15 token), la frase con maggior probabilità, calcolata con
# un modello di Markov di Ordine 1 e l'Add-One Smoothing

   frasi8Biden,frasi9Biden, frasi10Biden, frasi11Biden, frasi12Biden, frasi13Biden, frasi14Biden, frasi15Biden = SmistaLunghezzeFrasi_Markov(frasi1, lunghezzaCorpusBiden, tokensTOTbiden)
   frasi8OrdinatoBiden = OrdinaDizionario(frasi8Biden)
   frasi9OrdinatoBiden=OrdinaDizionario(frasi9Biden)
   frasi10OrdinatoBiden=OrdinaDizionario(frasi10Biden)
   frasi11OrdinatoBiden=OrdinaDizionario(frasi11Biden)
   frasi12OrdinatoBiden=OrdinaDizionario(frasi12Biden)
   frasi13OrdinatoBiden=OrdinaDizionario(frasi13Biden)
   frasi14OrdinatoBiden=OrdinaDizionario(frasi14Biden)
   frasi15OrdinatoBiden=OrdinaDizionario(frasi15Biden)
   
   print("Biden:")
   print()
   print("Frasi con probabilità maggiore:")
   print()
   print("Frase con lunghezza 8:")
   for key in frasi8OrdinatoBiden[:1]:
      print (key)
   
   print()
   print("Frase con lunghezza 9:")
   for key in frasi9OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 10:")
   for key in frasi10OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 11:")
   for key in frasi11OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 12:")
   for key in frasi12OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 13:")
   for key in frasi13OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 14:")
   for key in frasi14OrdinatoBiden[:1]:
      print (key)

   print()
   print("Frase con lunghezza 15:")
   for key in frasi15OrdinatoBiden[:1]:
      print (key)
   
   frasi8Trump,frasi9Trump, frasi10Trump, frasi11Trump, frasi12Trump, frasi13Trump, frasi14Trump, frasi15Trump = SmistaLunghezzeFrasi_Markov(frasi2, lunghezzaCorpusTrump, tokensTOTtrump)
   frasi8OrdinatoTrump =OrdinaDizionario(frasi8Trump)
   frasi9OrdinatoTrump=OrdinaDizionario(frasi9Trump)
   frasi10OrdinatoTrump=OrdinaDizionario(frasi10Trump)
   frasi11OrdinatoTrump=OrdinaDizionario(frasi11Trump)
   frasi12OrdinatoTrump=OrdinaDizionario(frasi12Trump)
   frasi13OrdinatoTrump=OrdinaDizionario(frasi13Trump)
   frasi14OrdinatoTrump=OrdinaDizionario(frasi14Trump)
   frasi15OrdinatoTrump=OrdinaDizionario(frasi15Trump)
   
   print()
   print("Trump:")
   print()
   print("Frasi con probabilità maggiore:")
   print()
   print("Frase con lunghezza 8:")
   for key in frasi8OrdinatoTrump[:1]:
      print (key)
   
   print()
   print("Frase con lunghezza 9:")
   for key in frasi9OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 10:")
   for key in frasi10OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 11:")
   for key in frasi11OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 12:")
   for key in frasi12OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 13:")
   for key in frasi13OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 14:")
   for key in frasi14OrdinatoTrump[:1]:
      print (key)

   print()
   print("Frase con lunghezza 15:")
   for key in frasi15OrdinatoTrump[:1]:
      print (key)
    
   print()
   
   print("**********")

# d) dopo aver individuato e classificato le Entità Nominate nel testo, stampare 
# i 15 nomi di persona più frequenti ordinati per frequenza
# 15 nomi di luogo più frequenti ordinati per frequenza

   personBiden, geoPEbiden, orgBiden= NamedEntity_tag (tokensPOStotBiden)
   personBidenOrdinato= OrdinaDizionario(personBiden)
   geoPEbidenOrdinato= OrdinaDizionario(geoPEbiden)
   orgBidenOrdinato= OrdinaDizionario(orgBiden)
   print ()
   print("Biden:")
   print()
   print("Entità nominate usate:")
   print("Nomi di persona usati: ")
   print()
   print(personBidenOrdinato)
   print()
   print("Nomi di luogo usati:")
   print()
   print(geoPEbidenOrdinato)
   print()
   print("Nomi di organizzazioni usate:")
   print()
   print(orgBidenOrdinato)
   print()
   print("15 nomi propri di persona più usati:")
   print("frequenza:\t nome:")
   for persona in personBidenOrdinato[:15]:
      print (persona[1], "\t\t" , persona[0][0]) 
   print()
   
   print("15 nomi propri di luogo più usati:")
   print("frequenza:\t nome:")
   for luogo in geoPEbidenOrdinato[:15]:
      print (luogo[1], "\t\t" , luogo[0][0])
   print()

   personTrump, geoPEtrump, orgTrump= NamedEntity_tag (tokensPOStotTrump)
   personTrumpOrdinato= OrdinaDizionario(personTrump)
   geoPEtrumpOrdinato= OrdinaDizionario(geoPEtrump)
   orgTrumpOrdinato= OrdinaDizionario(orgTrump)
   print ()
   print("Trump:")
   print()
   print ("Entità nominate usate:")
   print()
   print("Nomi di persona usati: ")
   print()
   print(personTrumpOrdinato)
   print()
   print("Nomi di luogo usati:")
   print()
   print(geoPEtrumpOrdinato)
   print()
   print("Nomi di organizzazioni usate:")
   print()
   print(orgTrumpOrdinato)
   print()
   
   print()
   print("15 nomi propri di persona più usati:")
   print("frequenza:\t nome:")
   for persona in personTrumpOrdinato[:15]:
      print (persona[1], "\t\t" , persona[0][0]) 
   print()
   
   print("15 nomi propri di luogo più usati:")
   print("frequenza:\t nome:")
   for luogo in geoPEtrumpOrdinato[:15]:
      print (luogo[1], "\t\t" , luogo[0][0])
   print()
   print("**********")


main(sys.argv[1], sys.argv[2]) 