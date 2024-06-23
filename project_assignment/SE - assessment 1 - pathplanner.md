# Assessment 1

Dit assessment bestaat uit twee delen. In het eerste deel vragen we je een zelf-gemaakte oplossing toe te lichten. Dit zou weinig voorbereiding moeten kosten. Het tweede deel is een programmeer opdracht gebaseerd op de drone zwerm die bij Avalor AI gebouwd wordt.

Als je vragen hebt kun je ons gerust bellen of mailen.

## Opdracht 1: Technische uitleg

Als software-enthousiastelingen hebben we vaak allemaal wel leuke hobby projecten gebouwd, gave algoritmes geschreven of iets anders moois gebouwd. Graag horen we je toelichting over een technisch software project of oplossing waar je zelf enthousiast over bent. Dat kan een persoonlijk hobby project zijn dat je hebt gebouwd, maar kan ook een technische oplossing zijn voor een professioneel probleem (dat je in zoverre kan delen). Graag horen we waarom je hier zo enthousiast over bent, welke afwegingen je hebt gemaakt en wat je er van hebt geleerd. Je kiest zelf hoe je het aan ons presenteert, maar onze voorkeur gaat uit naar visueel/schematisch en niet in regels code. Zorg dat je het in 5 minuten kan presenteren/toelichten.

We kijken naar je technische uitleg, en tegelijkertijd is dit een kans om elkaar wat beter te leren kennen. We beoordelen niet of dit project een technisch raakvlak heeft met het werk bij Avalor AI. 

Tijdens de assessment geef je ons je toelichting, waarbij wij wat vragen zullen stellen. Duur: +/- 20 min.

## Opdracht 2: Technische opdracht

Avalor AI werkt aan een geavanceerde drone zwerm die zelfstandig een gebied onder waarneming kan houden. Om een gebied effectief te kunnen doorzoeken maken wij gebruik van een wereldmodel. Een voorbeeld hiervan is een grid, verdeeld in gelijke vlakken, waarbij ieder vlak in het grid wordt voorzien van een numerieke waarde. Deze waarde representeert bijvoorbeeld de informatie die in dat vak aanwezig is. Wanneer een drone een pad vliegt, telt hij de waardes van alle vakken die hij passeert op, dit is dan de totaal waarde van het pad. Hoe langer je ergens niet geweest bent, hoe groter de kans is dat de situatie verandert is. Dat kan je representeren door, nadat je een vak bezocht hebt, de waarde van het vak langzaam weer te laten oplopen tot de originele waarde. Pas je model en algoritme aan om hiermee om te kunnen gaan.

### Opdracht

Schrijf een padplanningsalgoritme dat een zo waardevol mogelijk pad terug geeft voor een opgegeven N, t, T, (x, y).

Gegeven zijn:

- N: Grootte van het te doorzoeken grid: N x N.
- t: Totaal aantal discrete tijdstappen
- T: Maximale tijdsduur van het algoritme in milliseconden.
- (x, y): Index van startpositie van de drone

Alle waardes zijn discrete getallen.

Daarbij geldt:

- Het kost een drone 1 tijdstap om zich te verplaatsen naar een aangelegen vak. Dat kan horizontaal, verticaal en diagonaal.
- Het hele grid is zichtbaar voor de drone.
- De aanwezigheid van een drone in het vak resulteert tot het behalen van de waarnemingscore van het betreffende vlak. De drone kan per run van het algoritme meerdere keren de score van een vak behalen, aangezien het oploopt over de tijd. 
- Het algoritme moet altijd eindigen in de gegeven tijdlimiet T.
- De initiële score van elk vakje is gegeven in de data, maar de snelheid waarmee de score oploopt kan zelf gekozen worden. 

### Voorwaarden/tips:

- Je bent vrij in het kiezen van een programmeertaal.
- Schrijf het algoritme voor een enkele drone. Je hoeft voor nu geen rekening te houden met functionaliteit voor een drone-zwerm.
- Je algoritme dient binnen tijd T het pad plus de totaal behaalde score terug te geven.
- Schrijf een oplossing die ook werkt bij grotere waardes van N en t binnen de beschikbare tijd T.
- In de bijlage zijn drie voorbeeld levels van verschillende groottes meegegeven die je kan gebruiken.
- We kijken niet naar de uiteindelijke score, maar naar je methodiek, verantwoording en kwaliteit van de oplossing.

### Verdieping/uitdaging:

Mocht je wat extra uitdaging zoeken dan kun je de volgende challenge pakken:

- Zwerm algoritme: In werkelijkheid vliegen we natuurlijk met meerdere drones tegelijkertijd. Daarbij heeft het geen meerwaarde als drones het zelfde vak doorzoeken. Pas je model en algoritme aan om hiermee om te kunnen gaan.
  - De drones starten op willekeurige locaties in het grid.
  - Het algoritme moet om kunnen gaan met X aantal drones. Als input krijg je X verschillende start_posities en als resultaat geeft het X aantal paden en 1 totaal score terug.
  - Gebruik je eigen fantasie om benodigde aannames zelf te maken.

### Oplevering en beoordeling

Graag ontvangen we uiterlijk een uur voor het begin van de assessment een link naar je projectcode. Tijdens de assessment praat je ons door je code heen en zullen we wat vragen stellen. Duur: +/- 40 min.

Tijdens de assessment wordt er gekeken naar je code kwaliteit, je uitleg + toelichting op het algoritme, en je antwoorden op vervolg-vragen.
