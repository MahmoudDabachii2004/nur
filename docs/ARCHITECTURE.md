# NUR (نور) — Architecture & Décisions Master Document

> **نور** — Lumière en arabe. Concept coranique profond (Sourate An-Nur, 24:35).  
> Ce document capture CHAQUE décision, CHAQUE raison, CHAQUE référence de recherche.  
> Son but : permettre à quiconque de comprendre le projet dans son intégralité et de le reconstruire from scratch.

---

## Table des Matières

1. [Vision du Projet](#1-vision-du-projet)
2. [Analyse de Daleel.app — L'Inspiration](#2-analyse-de-daleelapp)
3. [Analyse Critique des Projets Existants](#3-analyse-critique-des-projets-existants)
4. [Les 7 Péchés Capitaux des Projets Existants](#4-les-7-péchés-capitaux)
5. [Architecture NUR — Les 10 Piliers](#5-architecture-nur---les-10-piliers)
6. [Sources de Données — Tout est Gratuit](#6-sources-de-données)
7. [Modèles LLM — Stratégie Hybride](#7-modèles-llm)
8. [Modèles d'Embedding](#8-modèles-dembedding)
9. [Vector Database](#9-vector-database)
10. [Techniques RAG Avancées](#10-techniques-rag-avancées)
11. [Anti-Hallucination — Pipeline de Vérification](#11-anti-hallucination)
12. [Sources Clickables — URLs](#12-sources-clickables)
13. [Avis des Savants — Obligation Éthique](#13-avis-des-savants)
14. [Ikhtilaf Awareness — Conscience du Désaccord](#14-ikhtilaf-awareness)
15. [Bilingual Strategy — Arabic-First](#15-bilingual-strategy)
16. [Débat Framework — LangChain vs Custom](#16-débat-framework)
17. [Contraintes Hardware — MacBook M4 16GB](#17-contraintes-hardware)
18. [Évaluation & Métriques](#18-évaluation--métriques)
19. [Recherche Académique — Papiers Clés](#19-recherche-académique)
20. [Projets Open-Source — Analyse Détaillée](#20-projets-open-source)
21. [Matrice Comparative Finale](#21-matrice-comparative)
22. [Appendice — Toutes les Références](#22-appendice)

---

## 1. Vision du Projet

### But
Créer un chatbot islamique qui répond à TOUTE question sur la religion en se basant EXCLUSIVEMENT sur le Coran, la Sunnah authentique, et les avis de savants reconnus. Chaque réponse DOIT citer ses sources de manière vérifiable et clickable.

### Principes Non-Négociables
- **0$ de dépense** — tout doit être gratuit
- **Fiabilité absolue** — une mauvaise réponse religieuse est pire que pas de réponse
- **Transparence totale** — chaque affirmation est traçable jusqu'à sa source
- **Avis des savants obligatoires** — les laymen ne peuvent pas interpréter les versets/hadiths seuls ; le système doit TOUJOURS rapporter les avis de savants qualifiés, jamais donner son propre avis
- **Arabic-first** — le texte arabe est la source de vérité, les traductions sont de l'aide à la compréhension
- **Francophone natif** — support complet français + arabe + anglais (avantage concurrentiel, aucun compétiteur ne le fait)
- **Respect du gradings hadith** — les hadiths da'if et mawdu' ne sont JAMAIS présentés comme preuve

### Pourquoi ce projet existe
Des millions de musulmans francophones n'ont pas accès facile à des sources authentiques. Les réseaux sociaux propagent des hadiths fabriqués. Daleel.app existe mais est premium et anglophone uniquement. Aucun outil gratuit, fiable, et francophone n'existe.

---

## 2. Analyse de Daleel.app

### Ce qu'est Daleel
- Application mobile iOS/Android, premium/freemium
- Tagline : "Trained strictly on Quran and Sunnah and built for a generation overwhelmed by misinformation"
- Fonctionnalités : vérification de hadiths, recherche de versets, vérification de contenu réseaux sociaux
- TOUJOURS fournit les sources — c'est leur différenciateur principal

### Comment ça fonctionne (inféré, car closed-source)
- Architecture RAG (Retrieval-Augmented Generation) — standard de l'industrie pour QA avec sources
- LLM probablement GPT-4 ou Claude (qualité des réponses)
- Base de données vectorielle contenant Coran + Hadiths indexés
- Sources TOUJOURS citées car le LLM ne répond qu'à partir du contexte récupéré

### Compétiteurs similaires
- **Tasleem AI** (tasleem.ai) — "Islamic Super App", 100K+ sources vérifiées
- **Deen Buddy** (deenbuddyapp.com) — AI Quran Chat
- **My Quran** (myquran.online) — Chatbot gratuit basé sur le Coran
- **Tarteel AI** (tarteel.ai) — Compagnon Coran, mémorisation, 15M+ utilisateurs

### Ce que Daleel fait bien (à copier)
- Toujours citer les sources
- Interface claire et simple
- Vérification de hadiths
- Contenu réseaux sociaux vérifiable

### Ce que Daleel ne fait PAS (nos opportunités)
- Pas de support français
- Pas d'avis des savants / ikhtilaf
- Premium (pas gratuit)
- Pas de mode offline
- Pas de grading hadith visible
- Pas de liens clickables vers les sources originales

---

## 3. Analyse Critique des Projets Existants

### 3.1 Hadith-AI (ahmedeltaher/Hadith-AI)

**Description** : Pipeline RAG pour hadiths en arabe avec LlamaIndex + Qdrant + Ollama

**Ce qui est bien** :
- Architecture modulaire propre — chaque module a une responsabilité unique (config, document_loader, embeddings, index_builder, query_engine)
- Configuration Pydantic avec override par variables d'environnement — pattern professionnel
- Deux stratégies de chunking disponibles : SentenceWindowNodeParser et SemanticSplitterNodeParser
- Vérification du modèle Ollama au démarrage — fail-fast avec message clair
- Schéma de données JSON avec métadonnées riches (narrateur, grade, book_number, chapter)
- Détection automatique de la collection hadith depuis le chemin du fichier
- CLI soignée avec Typer + Rich

**Ce qui est mal — CRITIQUE** :
- **ZÉRO prompt engineering** — la question brute est passée directement au LLM sans system prompt, sans instructions de citation, sans garde-fous. C'est le défaut le plus grave du projet
- **Reranker anglais sur texte arabe** — utilise `cross-encoder/ms-marco-MiniLM-L-12-v2` (modèle anglais) pour reranker du texte arabe. Résultat essentiellement aléatoire
- **Traductions anglaises ignorées dans le retrieval** — le schéma de données inclut un champ `english` mais seul le champ `text` (arabe) est indexé. L'anglais finit dans des métadonnées génériques jamais recherchées
- **Chunking naif pour les hadiths** — SentenceWindowNodeParser utilise une tokenisation de phrases english-centric. Un hadith est une unité sémantique atomique (isnad + matn) qui ne devrait JAMAIS être découpé. CHUNK_SIZE=512 avec CHUNK_OVERLAP=50 peut fragmenter un hadith à travers les boundaries
- **Qdrant en mémoire par défaut** — toutes les vectors perdues au redémarrage
- **Embedding Ollama custom redondant** — 138 lignes de code custom alors que l'intégration officielle LlamaIndex existe déjà dans les requirements. Le wrapper custom traite les "batchs" séquentiellement, sans retry, sans connection pooling
- **Aucune normalisation arabe** — pas de stripping des diacritiques (tashkeel), pas de normalisation des variantes alef (أ/إ/آ/ا), pas de normalisation ya/alif-maksura. كِتَابٌ et كتاب ont des embeddings différents alors que c'est le même mot
- **Grading hadith ignoré** — les données incluent le grade (Sahih) mais le pipeline ne l'utilise jamais : pas de filtrage, pas d'affichage, pas d'instruction au LLM
- **Aucun cross-referencing** — pas de lien Coran↔Hadith, pas de lien inter-collections, pas de mécanisme pour détecter les hadiths "muttafaq 'alayh" (présents dans Bukhari ET Muslim)
- **Citation basique** — juste rank + score + snippet de 200 chars. Pas de format structuré avec collection/narrateur/grade
- **Pas de multilingue** — pas de détection de langue, pas de réponse dans la langue de la question
- **Pas d'évaluation** — zéro script de test, zéro métrique, zéro benchmark

**Score global** : Code quality 7/10, Islamic domain awareness 1/10

---

### 3.2 Islam-GPT (Shaheer66/Islam-GPT)

**Description** : Chatbot islamique sur Coran + Sahih Muslim avec Mistral 7B et RAG

**Ce qui est bien** :
- Architecture RAG correcte de base (ingest → embed → store → retrieve → generate)
- Mistral-7B-Instruct-v0.3 — modèle open-weight approprié
- FAISS — léger et adapté pour un prototype
- 3 points d'entrée : CLI, class-based, Streamlit
- Index FAISS pré-construit inclus dans le repo

**Ce qui est mal — CRITIQUE** :
- **Projet forké d'un chatbot médical sans refactoring** — la classe s'appelle `SymptomPredictor`, le fichier `medibot.py`, des prompts médicaux dans les commentaires. Preuve que le domaine islamique n'a pas été pensé sérieusement
- **Embedding anglais pour texte religieux** — `all-MiniLM-L6-v2` (384 dimensions, optimisé anglais). Performances médiocres sur la terminologie islamique, les translittérations arabes, les nuances sémantiques religieuses
- **Chunking destructif** — `RecursiveCharacterTextSplitter` avec chunk_size=500, overlap=50. Les versets coraniques et les hadiths sont découpés par comptage de caractères. Un hadith peut être coupé en plein milieu de sa chaîne de narration (isnad)
- **Coran + Hadith mélangés dans le même index** — aucune métadonnée pour distinguer la parole d'Allah de la parole du Prophète. Le système ne peut pas différencier une révélation divine d'une tradition prophétique
- **Température 1.5 pour du QA religieux** — dangereusement élevé. Augmente massivement le risque d'hallucination. Un système religieux factuel devrait utiliser 0.1-0.3
- **max_length passé comme string** — `"512"` au lieu de `512`, probablement ignoré par l'API HuggingFace
- **Température inconsistante** — 1.5 dans certains fichiers, 0.5 dans d'autres
- **Pas de métadonnées de chunking** — aucun Surah, numéro de verset, numéro de hadith, narrateur attaché aux chunks. Attribution de source impossible après retrieval
- **Prompt minimal sans domaine islamique** — pas de persona, pas d'instruction de citation, pas de garde-fous, pas de distinction Coran/Hadith. Juste "Use the pieces of information provided in the context to answer user's question"
- **Citations source en raw Python** — `str(source_documents)` affiche le repr() des objets Document, inutilisable pour un utilisateur
- **Pas de support arabe** — anglais uniquement
- **Chemin d'index FAISS cassé** — le code référence `vectorstore/db_faiss` mais le repo contient `vector_store/index.pkl`
- **Pas de requirements.txt** — aucune gestion des dépendances
- **Pas de données source** — les PDFs Coran + Sahih Muslim ne sont pas inclus ni liés

**Score global** : Code quality 2/10, Islamic domain awareness 0/10

---

### 3.3 AraQA (Marje3na/AraQA)

**Description** : QA génératif arabe pour texte religieux authentique. IEE JAC-ECC 2023. Architecture Generate-Then-Verify (PAS RAG)

**Innovation clé** : Seul système avec vérification post-génération. Pipeline en 5 étapes : (1) 40,859 Q/A scrapées d'islamweb.net → nettoyées à 20K (2) Classification thématique par mots-clés (3) Fine-tune AraGPT2-Base (135M params) avec tokens spéciaux (4) Génération (5) Vérification post-hoc

**La vérification post-génération (seule existante)** :
- ~60 patterns regex arabes pour détecter les citations religieuses (قال تعالى, قال رسول الله صلى الله عليه وسلم, بقوله تعالى, etc.)
- Embedding des segments extraits + comparaison avec un dataset de référence authentique via `intfloat/multilingual-e5-base`
- Similarité cosinus : si > 0.90 → remplacer par le texte authentique ; si ≤ 0.90 → exclure entièrement
- Normalisation des diacritiques via `pyarabic.araby.strip_tashkeel` avant comparaison

**Ce qui est bien** :
- Seul système au monde avec vérification post-génération — innovation véritable
- Arabic-native — AraGPT2, multilingual-e5, pyarabic tous conçus pour l'arabe
- Publié et peer-reviewed avec évaluation formelle
- Normalisation des diacritiques avant similarité — sensé

**Ce qui est mal** :
- **Pas de RAG** — purement génératif avec vérification post-hoc. Pas de retrieval à l'inférence. Le modèle génère d'abord, puis on vérifie. L'approche RAG est fondamentalement plus sûre car le modèle ne génère qu'à partir de sources récupérées
- **Regex fragile** — 60+ patterns, cassants, manquent les formats de citation non-standards. Bug dans le code : paramètre de fonction nommé avec un mot répété ~20 fois, et référence de bracket manquante dans l'indexation
- **Modèle minuscule** — 135M params, 300 tokens max de génération. Insuffisant pour des réponses nuancées
- **Seuil binaire dur** — 0.90 est arbitraire, pas de scoring gradué de confiance
- **Pas d'évaluation de l'étape de vérification** — précision/rappel de l'extraction et du remplacement inconnus
- **Test statistiquement insignifiant** — perplexity 2.3 sur seulement 15 questions

**Leçon pour NUR** : La vérification post-génération est une idée brillante, mais l'implémentation par regex est fragile. On reprend le concept mais avec des méthodes plus robustes (NLI, vérification caractère par caractère, scoring de fidélité).

---

### 3.4 Quran RAG (Shariar-Mozumder/Quran_RAG)

**Description** : Chatbot RAG minimal pour le Coran avec Llama-3.2-1B

**Ce qui est bien** :
- Simplicité — facile à comprendre, dépendances minimales
- Double déploiement — CLI + Streamlit
- Beam search (num_beams=4, temp=0.5, rep_penalty=1.3) — paramètres de génération raisonnables pour QA factuel
- Prompt engineering explicite avec instruction de citer les versets

**Ce qui est mal** :
- **Mauvais embedding model** — `all-MiniLM-L6-v2` optimisé anglais (384-dim). Performances pauvres sur le contenu coranique arabe
- **Aucune vérification de source** — le modèle peut halluciner des numéros de versets
- **Pas de filtrage par métadonnées** — FAISS sans métadonnées, impossible de filtrer par Sourate ou thème
- **LLM de 1B params** — inadéquat pour du QA religieux nuancé
- **Index FAISS reconstruit à chaque démarrage** — pas de persistance
- **Extraction de réponse fragile** — recherche de string `result.find("Response:")`
- **Pas de mémoire de conversation**
- **Anglais uniquement**

---

### 3.5 HadithRAG (Quchluk/HadithRAG)

**Description** : RAG sémantique sur corpus hadith canonique, 50K+ hadiths de 9 ouvrages

**Ce qui est bien** :
- **Corpus le plus riche** — 50K+ hadiths de 9 ouvrages canoniques (Kutub al-Sittah + Musnad Ahmad, Muwatta Malik, Sunan al-Darimi)
- **30+ champs de métadonnées** — book, chapter, narrator, grade, subject, keywords, narrator_chain, related_hadiths
- **Affichage des documents source** — montre les hadiths sources avec métadonnées complètes pour vérification
- **Stockage efficace** — Parquet + ChromaDB pour 50K+ documents
- **Corpus bilingue** — arabe et anglais maintenus séparément
- **Pipeline modulaire** — séparation propre entre data prep, indexing, serving

**Ce qui est mal** :
- **Pas vraiment open-source** — nécessite une clé API OpenAI pour les embeddings ET la génération. Ne peut pas tourner offline
- **Embedding anglais uniquement** — `text_en` est le page_content pour l'indexation ; l'arabe est en métadonnée uniquement, non recherchable
- **Pas de retrieval sensible à l'authenticité** — a le champ `grade` mais ne filtre/ne pondère pas par Sahih/Hasan/Da'if
- **Pas de reranking** — similarité basique avec k=5
- **GPT-3.5-turbo** — obsolète, sujet aux hallucinations
- **Chemins hardcodés** — `/Users/Tosha/Desktop/...` dans le code
- **Pas de vérification de citation** — montre les sources mais ne vérifie pas que les affirmations du LLM correspondent

**Leçon pour NUR** : Leur schéma de métadonnées est le meilleur — on s'en inspire fortement. Mais on corrige le défaut critique : on utilise l'arabe comme texte principal pour l'embedding, pas l'anglais.

---

### 3.6 Quran-Hadith-Chatbot (hammadali1805)

**Description** : Pipeline RAG en deux étapes avec Gemini AI

**Innovation** : Query Expansion — utilise Gemini pour reformuler les requêtes avant le retrieval. C'est utile car les utilisateurs peuvent utiliser un langage colloquial. Exemple : "c'est quoi la règle pour le jeûne" → reformulé en "What are the Islamic rulings on fasting during Ramadan?"

**Ce qui est bien** :
- Query expansion — technique utile pour les requêtes islamiques en langage familier
- Vector store pré-construit — fonctionne out of the box
- ChromaDB persistant

**Ce qui est mal** :
- **Risque d'hallucination** — le prompt app.py dit "Summarise the given context" mais n'instruit PAS explicitement de n'utiliser que le contexte. La version use.py permet "existing knowledge" → hallucination garantie
- **Pas de citation structurée** — demande des "references" mais aucun mécanisme structuré
- **Pas de filtrage par métadonnées** — impossible de distinguer Coran vs Hadith
- **Pas de mémoire de conversation**
- **Pas de reranking**

---

### 3.7 IslamAI — oshoura (Islamicly)

**Description** : Full-stack RAG avec LangChain + Pinecone + Next.js. Le plus complet des projets analysés.

**Ce qui est bien** :
- **Meilleur anti-hallucination** — prompt explicite "don't know" + "only answer from sources"
- **Transparence des sources** — retourne et affiche TOUS les documents source via accordion UI
- **Reformulation de question domain-aware** — mapping explicite de terminologie islamique (sahaba = apostles, seerah = vie du Prophète)
- **Mémoire de conversation** — historique de chat passé pour condenser la question
- **Multi-source ingestion** — Coran + Hadith + Seerah avec metadata tagging
- **Disclaimer** — modal d'accueil : "should not be used for fatwa purposes... always double check by reviewing sources"
- **Scraper IslamQA** — collection innovante de données Q&A à grande échelle

**Ce qui est mal** :
- **Dépendance Pinecone** — service cloud payant ; pas d'alternative locale
- **Dépendance OpenAI** — embeddings et LLM sont OpenAI (vendor lock-in, coût)
- **Pas de reranking**
- **Tailles de chunk fixes** — 256 pour le Coran est agressif (peut couper des versets)
- **Pas de support arabe** — Coran anglais uniquement
- **Scraper IslamQA fragile** — scraper 500K URLs est cassant et éthiquement discutable

---

### 3.8 Fin-Islam (dannycahyo)

**Description** : Système multi-agent RAG pour la finance islamique. Le plus sophistiqué architecturalement.

**Architecture** : 4 agents en pipeline — Routing → Knowledge/Calculation → Compliance

**Innovation majeure — Compliance Agent** : Le SEUL projet avec validation Sharia post-génération. 5 règles : pas de promotion du Riba, pas de Gharar excessif, pas d'activités haram, terminologie respectueuse, concepts islamiques précis. 17 few-shot examples. Confidence threshold 0.7.

**Autres innovations** :
- Routing Agent classifie les requêtes en 6 catégories avec 20 few-shot examples
- Knowledge Agent avec fallback basé sur la confiance
- MCP (Model Context Protocol) pour calculs structurés Musharakah/Mudharabah
- Streaming SSE token par token
- Suite de tests complète (unit + integration + API)
- Docker Compose production + dev
- BasePromptBuilder avec 4 sections systématiques

**Ce qui est mal** :
- Domaine étroit — finance islamique uniquement, pas de Coran/Hadith
- Pas de support arabe
- Compliance agent peut être contourné — même LLM pour valider et générer
- Pas de reranking cross-encoder

**Leçon pour NUR** : Le Compliance Agent est une idée qu'on doit adapter — un "Islamic Guardrail Agent" qui vérifie que la réponse respecte les principes islamiques AVANT de la montrer à l'utilisateur.

---

### 3.9 Graph-Based RAG — Trustworthy Islamic Agent (maulanaanab)

**Description** : Dual Semantic + Graph RAG utilisant une ontologie coranique. L'approche la plus innovante architecturalement.

**Architecture** : Deux chemins de retrieval parallèles — Semantic RAG (standard) + Graph-Based RAG (ontologie). Résultats fusionnés avant génération.

**Le Graph RAG a 3 étapes** :
1. Concept Matching — embedding de la question comparé aux embeddings de concepts coraniques pré-calculés
2. Graph Traversal — pour chaque concept, traverse itérativement l'ontologie vers les concepts voisins les plus pertinents, collectant les versets associés
3. Verse Ranking — tri par similarité sémantique

**L'ontologie** — Sources du Quranic Arabic Corpus (corpus.quran.com) : 350 concepts (285 collectés), structurés en arbre avec description, sous-catégories, related_concepts, relevant_verses

**Innovation standout — Verse Embedding Enhancement** : Les versets ne sont PAS embeddés isolément. Chaque embedding inclut : N versets adjacents pour contexte + infos surah-level (nom, signification, nombre de versets, ordre de révélation, Mecquois/Médinois) + format de mapping direct

**Évaluation** : Surpasse GPT-4o-mini sur Islamic QA Dataset avec ROUGE et METEOR. Évaluation humaine : "Le Prophète Idris est-il mort?" → ChatGPT a récupéré 1 verset (mauvaise conclusion), DeepSeek 1 verset (inconclusif), Islamic Agent plusieurs versets (21:85-86 + 19:56-57, réponse nuancée correcte)

**Ce qui est mal** :
- Ontologie clairsemée — 285 concepts pour 6,236 versets, couverture extrêmement grossière
- Dépendance OpenAI
- Pas d'intégration hadith — noté comme futur travail mais essentiel
- Évaluation limitée — ROUGE/METEOR mesurent la similarité de surface, pas la précision factuelle
- Pas d'étude d'ablation — unclear combien le graphe ajoute vs semantic seul
- Pas de vérification post-génération

**Leçon pour NUR** : L'ontologie coranique est puissante pour le retrieval. On l'ajoute en Phase 4. Le verse embedding enhancement (contexte enrichi) est une meilleure pratique qu'on adopte directement.

---

### 3.10 IslamAI — yousefabuz17

**Description** : API REST Flask avec blueprint modulaire — PAS un chatbot, c'est une API de connaissances islamiques

**Points notables** :
- **Collection de données la plus riche** de tous les repos — 400+ fichiers : Coran (3 traductions × 114 sourates), 7 livres de hadiths, tafsir, lois islamiques, histoires prophétiques, duas, timelines
- Architecture modulaire avec singleton pattern, LRU caching, asyncio
- Suivi de crédibilité des sources avec documentation explicite
- Support de traduction via MarianMT
- Analyse de sentiment (unique)

**Pourquoi pas adapté** : Pas de LLM, pas de RAG, pas de QA. C'est une API de données, pas un assistant intelligent.

---

### 3.11 Chatbot-QnA-Quran (Bayhaqieee)

**Description** : Projet de recherche comparant T5, BART, Pegasus fine-tunés sur dataset synthétique de raisonnement coranique (857 paires Q/A avec Chain-of-Thought)

**Innovation** : Chain-of-Thought pour QA islamique — le dataset inclut `Complex_CoT` qui force le modèle à raisonner avant de répondre. Excellent concept pour les questions islamiques complexes.

**Pourquoi pas adapté** : Pas de chatbot fonctionnel, pas de RAG, pas de citation, juste des notebooks de recherche.

---

## 4. Les 7 Péchés Capitaux des Projets Existants

### Péché 1 : Chunking destructif
TOUS les projets découpent le texte sacré par comptage de caractères/tokens. Un verset coranique peut être coupé en deux. Un hadith peut avoir son isnad (chaîne de narration) séparé de son matn (contenu). C'est inacceptable pour du texte religieux où chaque unité est sémantiquement complète et indivisible.

### Péché 2 : Zéro prompt engineering islamique
La majorité des projets passent la question brute au LLM sans system prompt. Pas de définition du rôle, pas d'instructions de citation, pas de garde-fous contre les erreurs religieuses, pas de distinction Coran vs Hadith, pas de mention du grading, pas d'avertissement sur les fatwas.

### Péché 3 : Embedding inadapté
La plupart utilisent `all-MiniLM-L6-v2` (optimisé anglais, 384 dimensions) pour du texte qui contient de l'arabe et de la terminologie islamique. C'est comme utiliser un dictionnaire espagnol pour traduire du japonais — le matching sémantique est fondamentalement mauvais.

### Péché 4 : Grading hadith ignoré
Les données ont les grades d'authenticité (sahih, hasan, da'if) mais aucun projet ne les utilise dans le pipeline. Pas de filtrage, pas de pondération, pas d'affichage. C'est irresponsable pour du contenu religieux où la différence entre un hadith sahih et un hadith da'if est fondamentale.

### Péché 5 : Pas de vérification post-génération
Sauf AraQA (avec une implémentation fragile par regex), aucun projet ne vérifie que les citations du LLM correspondent réellement aux sources. Le LLM peut inventer des versets coraniques ou attribuer des hadiths à la mauvaise collection, et personne ne le vérifie.

### Péché 6 : Monolingue
Tous les projets sont soit anglais-only, soit arabe-only. Aucun ne gère correctement le multilingue. Aucun ne supporte le français. Pourtant, des centaines de millions de musulmans sont francophones.

### Péché 7 : Coran et Hadith mélangés
Plusieurs projets indexent le Coran et les hadiths dans le même espace vectoriel sans métadonnées de type. Le système ne peut pas distinguer la parole d'Allah (Coran) de la parole/acte du Prophète ﷺ (Hadith). Cette distinction est théologiquement fondamentale.

---

## 5. Architecture NUR — Les 10 Piliers

### Pilier 1 : Architecture Triple-Index (+ 1 Index Savants)

Au lieu d'un seul index mixte, on utilise **4 collections séparées** dans la base vectorielle :

**INDEX CORAN** — 6,236 ayahs, chunk par ayah, texte arabe comme source primaire
**INDEX HADITH** — 50K+ hadiths, chunk par hadith individuel, texte arabe (+ anglais quand disponible)
**INDEX TAFSIR** — Tafsir Ibn Kathir et autres, chunk par section de commentaire (1-3 ayahs)
**INDEX SAVANTS** — Fatwas, avis de fiqh, opinions savantes, chunk par fatwa/avis

Chaque index a son propre schéma de métadonnées adapté. Les 4 indexes sont liés par une **Cross-Reference Database** qui mappe :
- Ayah → Hadiths qui l'expliquent/commentent (via tafsir)
- Hadith → Versets coraniques qu'il référence
- Fatwa/Savants → Preuves coraniques et hadithiques sur lesquelles l'avis est basé
- Ayah → Tafsir qui la commente

**Pourquoi c'est mieux** :
- Recherche ciblée : on peut chercher SEULEMENT dans le Coran si on veut
- Filtrage par grade d'authenticité côté hadith
- On sait TOUJOURS si une réponse vient du Coran, d'un hadith, du tafsir, ou d'un savant
- Le LLM peut citer correctement : "Dans le Coran, Allah dit..." vs "Le Prophète ﷺ a dit..." vs "L'Imam Malik a dit..."
- L'Index Savants permet de rapporter les avis qualifiés, pas d'inventer

---

### Pilier 2 : Arabic-First Cross-Lingual Retrieval

**Principe** : Le texte arabe est la source de vérité. Les traductions sont de l'aide à la compréhension, pas des sources primaires. On n'embedde PAS de traduction comme source.

**Le problème des traductions** : Les traductions du Coran en français/anglais sont des interprétations humaines, imparfaites par nature. Un même mot arabe peut être traduit différemment selon le traducteur. Embedder une traduction imparfaite, c'est embedder une imperfection dans le système.

**L'approche** : Les modèles d'embedding multilingues comme `BAAI/bge-m3` sont entraînés pour le cross-lingual — ils placent "zakat on gold" (anglais), "zakat sur l'or" (français), et "زكاة الذهب" (arabe) dans le même espace vectoriel. Le matching se fait DIRECTEMENT entre la langue de la question et l'arabe, sans étape de traduction intermédiaire.

**Bénéfices** :
- Zéro latence de traduction
- Zéro perte de précision due à la traduction
- Le texte arabe original est toujours la source de vérité
- L'utilisateur voit l'arabe + sa traduction pour compréhension, mais c'est l'arabe qui fait foi

**Pour les hadiths** : Priorité à l'embedding du texte arabe quand disponible. Pour les collections uniquement disponibles en anglais, embedding anglais en fallback. Jamais de traduction FR comme source primaire.

---

### Pilier 3 : Authenticity-Weighted Retrieval

Au lieu de simplement exclure les hadiths faibles, on **pondère le score de similarité** par le grade d'authenticité :

**Sahih** : +30% boost — on fait confiance à la chaîne de narration
**Hasan** : +10% boost — acceptable mais chaîne légèrement plus faible
**Da'if** : -50% penalty — dernier recours, seulement si aucune autre source
**Mawdu'** : EXCLU — fabriqué, jamais montré à l'utilisateur

**Logique** : Un hadith sahih moins pertinent en similarité peut remonter au-dessus d'un hadith da'if plus "similaire". C'est éthiquement plus responsable pour du contenu religieux. En islam, la force de la chaîne de narration est plus importante que la similarité textuelle.

**Affichage obligatoire** : Chaque hadith cité doit afficher son grade. Si c'est hasan ou da'if, un avertissement doit accompagner la citation.

---

### Pilier 4 : Vérification Post-Génération (Anti-Hallucination)

Pipeline en 3 étapes après que le LLM génère sa réponse :

**Étape 1 — Extraction des citations** : Identifier toutes les références à des versets coraniques et hadiths dans la réponse du LLM. Méthode : regex avancé + NLI (Natural Language Inference) pour les citations implicites.

**Étape 2 — Vérification caractère par caractère (Coran)** : Pour TOUT verset coranique cité dans la réponse, comparer le texte cité contre le texte officiel arabe caractère par caractère. Le Coran est la parole d'Allah — on ne tolère AUCUNE erreur. Pour les hadiths : vérifier que la référence (collection + numéro) existe réellement, et comparer le texte cité contre le texte authentique.

**Étape 3 — Score de fidélité** : Chaque affirmation factuelle dans la réponse doit être traçable à un des documents récupérés. Si une affirmation ne peut être tracée → flag comme hallucination potentielle. Score = (affirmations soutenues) / (affirmations totales).

**Actions selon le score** :
- Score élevé → Afficher la réponse avec confiance
- Citation fausse détectée → Remplacer par le texte authentique + avertissement
- Trop d'hallucinations → "Je ne peux pas répondre avec confiance à partir des sources disponibles"

**Inspiré de** : AraQA (concept de vérification post-génération) mais avec des méthodes robustes au lieu de regex fragile, et RAG au lieu de génération pure.

---

### Pilier 5 : Context-Enriched Chunks (Contextual Retrieval d'Anthropic)

Technique découverte par Anthropic en 2024 : préfixer chaque chunk avec du contexte généré par LLM. Impact : réduction de 67% des échecs de retrieval.

**Problème** : Un chunk nu comme "Zakat is 2.5% of wealth held for one lunar year" manque de contexte. L'embedding ne sait pas si c'est du Coran, un hadith, du fiqh, quel thème, quelles cross-references.

**Solution** : Avant l'embedding, chaque chunk est enrichi :
- Pour un verset coranique : "Ce verset est de Sourate Al-Baqarah, ayah 255, connu comme Ayat al-Kursi. Il traite de l'omnipotence et omniscience d'Allah. Révélation mecquoise. Lié aux hadiths sur le mérite de la récitation."
- Pour un hadith : "Ce hadith est de Sahih Bukhari #1234, Livre de la Zakat, chapitre sur l'obligation. Narré par Abu Hurairah. Grade : Sahih. Lié au Coran 2:177, 9:60."
- Pour un avis savant : "Cet avis est de l'Imam Malik dans Al-Muwatta, chapitre de la prière. Il concerne le temps de la prière du Dhuhr. École Maliki. Diffère de l'avis Hanafi sur..."

**Pourquoi c'est transformateur** : Un utilisateur qui demande "aumône obligatoire" peut matcher "zakat" grâce au contexte sémantique riche du chunk. Sans ce contexte, le matching dépend uniquement du modèle cross-lingual.

---

### Pilier 6 : LLM Hybride — Groq First + Ollama Fallback

**Groq (Cloud — Priorité)** :
- Vitesse : 450+ tok/s pour Llama 3.3 70B — réponses quasi-instantanées
- Qualité : Llama 3.3 70B est excellent pour la compréhension et génération
- Coût : GRATUIT (6K TPM sur le tier gratuit)
- Idéal pour : requêtes normales, conversations, questions complexes

**OpenRouter — Qwen3-32B (Cloud — Requêtes arabes)** :
- Meilleur support arabe que Llama
- 27+ modèles gratuits sur OpenRouter
- Idéal pour : requêtes en arabe, questions nécessitant une compréhension arabe fine

**Ollama — Qwen2.5-7B (Local — Fallback/Offline/Privacy)** :
- 100% local sur le MacBook M4
- ~15-18 tok/s — plus lent mais fonctionnel
- Support arabe excellent (Qwen est chinois, optimisé multilingue)
- Idéal pour : mode offline, questions privées, quand les APIs sont down
- Utilise Apple MLX backend (93% plus rapide que llama.cpp sur Apple Silicon)

**Principe clé** : Les embeddings tournent 100% en local — les textes sacrés ne quittent jamais la machine. Seule la question reformulée + le contexte récupéré va au cloud LLM pour la génération.

---

### Pilier 7 : Structured Citation Protocol (Source ID Injection)

Au lieu d'espérer que le LLM cite correctement, on **injecte des IDs de sources numérotées** dans le prompt et on **exige** que le LLM les utilise.

**Fonctionnement** : Les documents récupérés sont présentés au LLM avec des identifiants :
- [S1] Coran - Al-Baqarah 2:255 (Arabe) — Grade: N/A (Coran, parole d'Allah)
- [S2] Sahih Bukhari #1234 - Abu Hurairah — Grade: Sahih
- [S3] Tafsir Ibn Kathir - Commentaire de 2:255

**Règle dans le prompt** : "Chaque affirmation DOIT citer [S1], [S2], ou [S3]. Ne cite PAS une source qui n'est pas listée ci-dessus."

**Post-traitement** : Après génération, on mappe chaque [SX] vers un affichage riche : texte arabe original, traduction, grade d'authenticité, lien clickable vers la source en ligne.

**Pourquoi c'est nécessaire** : Sans ce protocole, le LLM peut inventer des références ("Sahih Bukhari #99999" qui n'existe pas), attribuer un hadith à la mauvaise collection, ou paraphraser un verset en prétendant que c'est le texte exact. Le Source ID Protocol élimine ces risques.

---

### Pilier 8 : Avis des Savants — Obligation Éthique

**Pourquoi c'est non-négociable** : Un layman (non-savant) qui interprète un verset ou un hadith par lui-même risque :
- Mauvaise compréhension du contexte
- Commission de haram par ignorance
- Dans le pire cas : shirk (associationisme) par mauvaise interprétation d'attributs divins
- Bid'ah (innovation religieuse) en ajoutant des pratiques non fondées

**Règle absolue** : Le système ne DONNE JAMAIS son propre avis. Il rapporte TOUJOURS : "L'Imam X a dit..." avec la source du livre, et les preuves (Coran + Hadith) sur lesquelles l'avis est basé.

**Structure d'un avis savant** :
- Nom du savant et son époque
- École de jurisprudence (madhhab) si applicable
- L'avis lui-même, formulé précisément
- Source livresque (nom du livre, volume, page/chapitre)
- Preuves sur lesquelles l'avis est basé (versets + hadiths)
- S'il y a ikhtilaf (désaccord), mentionner les autres avis

**Sources de données pour les avis savants** (toutes gratuites) :
- IslamQA.info — fatwas avec citations Coran+Hadith+Savants, disponible en plusieurs langues dont l'arabe et l'anglais
- Islamweb.net — fatwas du Qatar, multi-langues, savants reconnus
- Dar al-Ifta (Egypte) — fatwas officielles d'une institution reconnue
- Shamela.ws — bibliothèque de livres classiques arabes (Al-Mughni, Al-Muhadhdhab, Al-Mabsout, etc.)
- IslamFatwa — fatwas classées par thème

**Priorité des sources savantes** :
1. Savants classiques (fondateurs des 4 écoles : Abu Hanifa, Malik, Shafi'i, Ahmad ibn Hanbal)
2. Commentateurs reconnus (Ibn Kathir, Ibn Hajar, Nawawi, etc.)
3. Savants contemporains reconnus (Ibn Baz, Uthaymeen, Albani, etc.)
4. Institutions de fatwa (Dar al-Ifta, IslamQA, etc.)

---

### Pilier 9 : Ikhtilaf Awareness — Conscience du Désaccord Savant

**Quoi** : Quand il existe un désaccord (ikhtilaf) entre les savants sur une question, le système le détecte et présente TOUS les avis avec leurs preuves respectives.

**Pourquoi** : Présenter un seul avis comme vérité unique est :
- Théologiquement incorrect — l'ikhtilaf est reconnu et respecté dans la tradition islamique
- Potentiellement trompeur — l'utilisateur croit qu'il n'y a qu'un avis
- Sectaire — privilégier une école sans mentionner les autres

**Comment le système gère l'ikhtilaf** :
- Si tous les savants sont d'accord → réponse simple avec consensus (ijma')
- S'il y a désaccord → présenter chaque avis avec : le savant/école, la preuve, et le raisonnement
- Toujours ajouter : "Cette question relève du fiqh — consulte un savant qualifié de ton école pour une réponse définitive"
- Ne JAMAIS prendre position ou dire quel avis est "meilleur"

**AUCUN projet existant ne fait cela**. C'est probablement la feature qui démarque le plus NUR de tout ce qui existe.

---

### Pilier 10 : French-First Multilingual

**Pourquoi le français est un avantage concurrentiel** :
- Des centaines de millions de musulmans francophones : Maroc, Algérie, Tunisie, France, Sénégal, Mali, Niger, Guinée, Comores, Djibouti, etc.
- AUCUN chatbot islamique sérieux ne supporte le français nativement
- Daleel = anglais, Hadith-AI = arabe, tous les autres = anglais
- C'est le marché le plus large et le moins servi

**Traduction du Coran en français** : Principalement la traduction de **Muhammad Hamidullah** — disponible via fawazahmed0/quran-api. C'est la traduction la plus respectée en français.

**Support trilingue** :
- **Arabe** : langue source, texte sacré, toujours affiché
- **Français** : langue principale de l'interface et des réponses
- **Anglais** : fallback, beaucoup de contenu hadith disponible uniquement en anglais

---

## 6. Sources de Données — Tout est Gratuit

### Coran

| Source | Ce qu'elle offre | Format | Rate Limit | Coût |
|--------|-----------------|--------|-----------|------|
| fawazahmed0/quran-api | 440+ traductions, 90+ langues, arabe Uthmani et IndoPak | REST API (CDN jsDelivr) | Aucun | Gratuit |
| alquran.cloud | Coran complet, tafsir, audio | JSON REST API | Aucun | Gratuit |
| QUL (Tarteel AI) | JSON/CSV/SQL téléchargeable offline | JSON/CSV/SQLite | N/A (téléchargement) | Gratuit |
| quran.com | API GraphQL officielle, word-by-word | GraphQL | Aucun | Gratuit |
| HuggingFace anisafifi/multilingual-quran | Arabe + EN + FR + Indonésien | HuggingFace Dataset | N/A | Gratuit |

**Traductions clés disponibles** :
- Arabe : Uthmani (texte sacré original) et IndoPak
- Français : Hamidullah (plus respectée), et autres
- Anglais : Saheeh International, Yusuf Ali, Pickthall, et 440+ autres

### Hadiths

| Source | Ce qu'elle offre | Grades | Format | Coût |
|--------|-----------------|--------|--------|------|
| fawazahmed0/hadith-api | Multi-langues, multi-grades, toutes collections | ✅ Sahih/Hasan/Da'if | REST API | Gratuit |
| AhmedBaset/hadith-json | 50,884 hadiths AR + EN | ✅ Inclus | JSON | Gratuit |
| sunnah.com API | Collections majeures, AR + EN | ✅ Inclus | REST API (clé requise) | Gratuit |
| sunnah.now | API moderne, scholarly-curated | ✅ Inclus | REST API | Gratuit |
| The9Books API | 9 livres canoniques | ✅ Inclus | REST API | Gratuit |
| HuggingFace fawazahmed0/hadith-data | Données hadith | ✅ Inclus | HuggingFace | Gratuit |
| HuggingFace meeAtif/hadith_datasets | 6 livres majeurs | ✅ Inclus | JSON/CSV | Gratuit |

**Collections couvertes** (Kutub al-Sittah + plus) :
- Sahih al-Bukhari : 7,008 hadiths
- Sahih Muslim : 5,362 hadiths
- Sunan Abu Dawud : 4,590 hadiths
- Jami at-Tirmidhi : 3,956 hadiths
- Sunan an-Nasai : 5,662 hadiths
- Sunan Ibn Majah : 4,341 hadiths
- Muwatta Malik, Musnad Ahmad, Sunan ad-Darimi

**Grades d'authenticité** :
- **Sahih** (Authentique) — Grade le plus élevé, chaîne de narration rigoureusement vérifiée
- **Hasan** (Bon) — Acceptable, chaîne légèrement plus faible
- **Da'if** (Faible) — Chaîne questionnable, pas utilisé pour les rulings
- **Mawdu'** (Fabriqué) — Définitivement fabriqué, ne doit JAMAIS être cité comme preuve

### Tafsir

| Source | Ce qu'elle offre | Format | Coût |
|--------|-----------------|--------|------|
| spa5k/tafsir_api | Tafsir multiple | REST API | Gratuit |
| QUL Tarteel | Ibn Kathir et autres | JSON/SQL | Gratuit |
| Kaggle Quran Tafsir Ibn Kathir | Ibn Kathir en anglais | JSONL | Gratuit |
| QuranAPI Tafsir | Ibn Kathir, Maarif Ul Quran | REST API | Gratuit |

### Avis de Savants / Fatwas

| Source | Ce qu'elle offre | Coût |
|--------|-----------------|------|
| IslamQA.info | Fatwas avec sources Coran+Hadith+Savants | Gratuit |
| Islamweb.net | Fatwas Qatar, multi-langues | Gratuit |
| Dar al-Ifta Egypte | Fatwas officielles | Gratuit |
| Shamela.ws | Bibliothèque de livres classiques arabes | Gratuit |

---

## 7. Modèles LLM — Stratégie Hybride (Vérifié par Recherche)

### Cloud LLM — Groq (Priorité #1, 100% Gratuit)

Groq est le provider principal. Tous ces modèles sont disponibles gratuitement sur le plan gratuit de Groq. Les évaluations ci-dessous sont basées sur des benchmarks indépendants, pas des opinions.

#### Modèles Groq — Évaluation Vérifiée

##### 🏆 Qwen 3 32B (qwen/qwen3-32b) — MEILLEUR CHOIX GLOBAL

**Faits vérifiés** :
- Release : Avril 2025, par Alibaba Qwen Team
- Architecture : Dense transformer, 32B params, 128K contexte
- 119 langues supportées dont arabe (standard + dialectes : Najdi, Levantine, Egyptian, Moroccan, etc.) et français
- Hybrid thinking mode (thinking + non-thinking)
- Function calling natif
- **OALL (Open Arabic LLM Leaderboard) v2** : Top 10, score moyen 75+ sur benchmarks arabes
- **AlGhafa** : 80.66 (leader sur ce benchmark)
- **MadinahQA** : compétitif
- Arabe officiellement supporté et entraîné sur des données arabes massives

**Limites** :
- Pas de vision (text-only)
- Pas de benchmark spécifique IslamicQA public
- Rapports communautaires : tendance à répondre en anglais quand questionné en français (à vérifier)

**Limites Groq gratuit** : 60 RPM, 1K RPD, 6K TPM, 500K TPD

**Pour NUR** : 🥇 Modèle principal pour les requêtes arabes ET françaises. Meilleur rapport qualité/arabe/multilingue sur Groq. 60 RPM = le plus de requêtes/minute.

---

##### 🥈 Qwen 3.6 27B (qwen/qwen3.6-27b) — MEILLEUR RAISONNEMENT + MULTIMODAL

**Faits vérifiés** :
- Release : Avril 2026, par Alibaba Qwen Team
- Architecture : Hybride Gated DeltaNet + Gated Attention (nouvelle archi, pas un transformer standard)
- 27B params dense, contexte natif 262K (131K sur Groq)
- **201 langues** supportées (vs 119 pour Qwen3-32B) — plus de dialectes arabes
- Nativement multimodal (text + image + vidéo)
- Thinking mode + Thinking Preservation (nouveau : garde le contexte de raisonnement entre les tours)
- MMLU-Pro : 86.2, GPQA Diamond : 87.8, AIME26 : 94.1 — **scores de raisonnement supérieurs à Qwen3-32B**
- SWE-bench Multilingual : 71.3 — bon multilingue

**Limites** :
- Problèmes FP8 : quantization FP8 produit du "gibberish" — à éviter
- OOM sur GPU consumer à cause du vision encoder
- Prompt cache cassé avec les couches DeltaNet (state recurrent perdu)
- 2x plus cher sur Groq que Qwen3-32B ($0.60 vs $0.29/M input)
- Vision peut ne pas être exposée sur l'API Groq (à vérifier)
- Rapports de language-mixing en français (répond en anglais)

**Limites Groq gratuit** : 30 RPM, 1K RPD, 8K TPM, 200K TPD

**Pour NUR** : 🥈 Utilisé pour les questions complexes de fiqh nécessitant du raisonnement profond (thinking mode). Sa capacité multimodale (lecture d'images de manuscrits, pages Coran) est un plus futur. Supérieur en raisonnement pur mais moins stable et plus cher que Qwen3-32B.

---

##### Llama 3.3 70B (llama-3.3-70b-versatile) — BON EN FRANÇAIS, FAIBLE EN ARABE

**Faits vérifiés** :
- Release : Décembre 2024, par Meta
- Architecture : Dense transformer, 70B params, 128K contexte
- **8 langues officiellement supportées** : anglais, allemand, français, italien, portugais, hindi, espagnol, thaï
- **⚠️ ARABE N'EST PAS OFFICIELLEMENT SUPPORTÉ** — Meta le dit explicitement : l'arabe ne rencontre pas leurs "performance thresholds for safety and helpfulness"
- OALL v2 : Rank #9 (74.47 moyen) — derrière TOUS les modèles Qwen du top 10
- **MadinahQA** : 80.36 — leader sur ce benchmark spécifique (grammaire arabe)
- MGSM : 91.1 (aggregate multilingual, incluant français) — bon
- MMLU-Pro : 68.9, MATH : 77.0 — correct
- "Versatile" = endpoint standard qualité sur Groq (vs "SpecDec" = speculative decoding 6x plus rapide)
- French IS officiellement supporté avec bons scores

**Limites** :
- Arabe non-supporté → hallucinations probables en arabe, tool-calling dégradé en arabe
- Safety filters over-triggering sur contenu religieux non-anglais
- Seulement 8 langues vs 119+ pour Qwen

**Limites Groq gratuit** : 30 RPM, 1K RPD, 12K TPM, 100K TPD

**Pour NUR** : ⚠️ Seulement pour les requêtes FRANÇAISES pures (sans arabe). Ne PAS utiliser pour du contenu arabe. Le français est officiellement supporté et de bonne qualité. 12K TPM = bon throughput.

---

##### Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct) — HAUT THROUGHPUT, VISION, MAIS CONTROVERSÉ

**Faits vérifiés** :
- Release : Avril 2025, par Meta
- Architecture : MoE, 16 experts, 2 actifs/token, **17B params actifs / 109B total**
- 12 langues supportées dont arabe et français
- Entraîné sur 200 langues (100+ avec >1B tokens chacune — 10x plus que Llama 3)
- Nativement multimodal (vision), contexte annoncé 10M (mais **131K seulement sur Groq**)
- DocVQA : 94.4 — excellent pour compréhension de documents
- MMLU-Pro : 74.3, GPQA Diamond : 57.2
- MGSM : 90.6 (vs 91.1 pour Llama 3.3 70B — légèrement inférieur)
- Vitesse : 460-625+ tok/s sur Groq

**Limites CRITIQUES** :
- **Controverse LMArena** : Meta a soumis une version différente pour le benchmark que celle évaluée en interne. LMArena a publié un statement disant que l'interprétation de Meta "ne correspondait pas à leurs attentes"
- **LiveBench (sans contamination)** : résultats pauvres — performance réelle inférieure aux benchmarks officiels
- **Communauté** : rapports de Scout "substantiellement pire que Llama 3.3 70B" en traduction
- Vision : anglais uniquement pour la compréhension d'images
- Taux d'hallucination : augmentation rapportée dans les mises à jour
- 109B params totaux doivent être chargés même si seulement 17B actifs → ~60GB RAM nécessaire en local

**Limites Groq gratuit** : 30 RPM, 1K RPD, 30K TPM, 500K TPD

**Pour NUR** : ⚠️ Seulement si on a besoin de throughput massif (30K TPM = 5x plus que Llama 3.3) ou de vision. Sinon, Qwen3-32B est meilleur pour l'arabe et plus fiable. La controverse de benchmark est un red flag.

---

##### GPT OSS 120B (openai/gpt-oss-120b) — FORT RAISONNEMENT ANGLAIS, FAIBLE MULTILINGUE

**Faits vérifiés** :
- Release : 5 août 2025, par OpenAI — leur premier modèle open-weight depuis GPT-2 (2019)
- Architecture : MoE, 128 experts, 4 actifs/token, **5.1B params actifs / 117B total**, 128K contexte
- License : Apache 2.0
- Entraîné sur dataset "majoritairement anglais, orienté STEM, code, et connaissances générales"
- **⚠️ FAIBLESSE MULTILINGUE DOCUMENTÉE** : papier arXiv 2508.12461v1 ("Is GPT-OSS Good?") note "notable weaknesses in multilingual tasks"
- **AUCUN benchmark arabe ou français publié** par OpenAI
- GPQA Diamond : 80.1 (approche o4-mini)
- Surpasse o4-mini sur AIME math et HealthBench
- Partenariat HUMAIN (Arabie Saoudite) et Orange (France) — mais pour développer leurs propres modèles, pas pour améliorer GPT-OSS lui-même
- 128K contexte, reasoning effort ajustable (Low/Medium/High)

**Limites CRITIQUES** :
- **LiveBench** : ranké 24ème, **en dessous de Qwen3-32B** qui est 4x plus petit
- **Benchmark overfitting** : chute de 15 positions entre benchmarks publics et LiveBench (sans contamination)
- **Inverse scaling** : GPT-OSS-20B bat le 120B sur plusieurs benchmarks (MMLU, HumanEval)
- **SimpleQA** : GPT-OSS-20B score 6.7/100 en exactitude factuelle
- **CoT non supervisé** : OpenAI recommande de ne PAS montrer le chain-of-thought aux utilisateurs (peut contenir des hallucinations)
- **Aucune donnée arabe/français** = risque élevé d'hallucination dans ces langues

**Limites Groq gratuit** : 30 RPM, 1K RPD, 8K TPM, 200K TPD

**Pour NUR** : ❌ Ne PAS utiliser. Faiblesse multilingue documentée, aucun benchmark arabe/français, risque d'hallucination élevé. Le raisonnement est bon en anglais mais inutile pour nous sans arabe et français fiables.

---

##### GPT OSS 20B (openai/gpt-oss-20b) — MÊME PROBLÈME QUE 120B, EN PLUS PETIT

Même architecture MoE que le 120B mais 21B total / 3.6B actifs, 32 experts, 128K contexte. Mêmes faiblesses multilingues. Score SimpleQA de 6.7/100 (connaissances factuelles catastrophiques). Bat parfois le 120B sur certains benchmarks (inverse scaling paradox).

**Pour NUR** : ❌ Même conclusion que le 120B. Pas adapté pour multilingue.

---

##### Llama 3.1 8B (llama-3.1-8b-instant) — RAPIDE ET PAS CHER, MAIS LIMITÉ

Dense 8B, 128K contexte. Même profil linguistique que Llama 3.3 (8 langues, pas d'arabe officiel). Le seul avantage : 14.4K RPD = le plus de requêtes par jour sur Groq. Qualité inférieure aux 70B+ pour du QA nuancé.

**Pour NUR** : ⚠️ Seulement pour des requêtes très simples ("combien de rak'ahs dans Fajr ?"). Pas adapté pour du fiqh ou des réponses avec sources.

---

#### Modèles Groq Spéciaux — Évaluation Vérifiée

##### Orpheus Arabic Saudi (canopylabs/orpheus-arabic-saudi) — TTS Arabe, MAIS PAS POUR LE CORAN

**Faits vérifiés** :
- Par Canopy Labs, base Llama-3B, codec SNAC, 24kHz WAV
- 6 voix saoudiennes (Abdullah, Fahad, Sultan, Lulwa, Noura, Aisha)
- Optimisé pour le dialecte saoudien, PAS l'arabe classique/coranique
- Update avril 2026 : moins d'hallucinations, meilleur handling des nombres

**⚠️ CRITIQUE — PAS adapté pour le Coran** :
- Dialecte saoudien ≠ arabe coranique (registre linguistique différent)
- **PAS de support tajweed** (madd, ghunnah, idgham, qalqalah, waqf)
- Les diacritiques améliorent la prononciation mais ne produisent PAS du tajweed
- **200 caractères max par requête** — la plupart des versets dépassent
- Pas de vocal directions (contrairement au modèle anglais)
- Prononciation coranique incorrecte = problème religieux sérieux

**Pour les hadiths** : ⚠️ Partiellement acceptable — le hadith ne requiert pas le tajweed, mais le dialecte saoudien ne correspond pas au registre classique attendu

**Pour NUR** : ❌ Coran : ne PAS utiliser. ⚠️ Hadiths : possible avec avertissement. ✅ Interface/UX : bon pour lire des instructions ou résumés en dialecte. Feature future intéressante mais avec limitations claires.

**Limites Groq gratuit** : 10 RPM, 100 RPD, 1.2K TPM, 3.6K TPD

---

##### Whisper Large v3 / v3 Turbo — Input Vocal

**Faits vérifiés** : Modèle de transcription OpenAI, support multilingue arabe/français/anglais, 20 RPM, 2K RPD sur le plan gratuit.

**Pour NUR** : ✅ Feature future — permettre à l'utilisateur de parler au lieu de taper. Combiné avec le LLM + Orpheus (pour la réponse), on aurait une conversation vocale partielle.

---

##### Prompt Guard 2-22M / 2-86M — Safety

Modèles de détection de prompts malveillants par Meta. 30 RPM, 14.4K RPD. Utile pour filtrer les tentatives de prompt injection ou de manipulation du système.

**Pour NUR** : ✅ À intégrer dans le pipeline pour protéger contre les tentatives de contourner les garde-fous islamiques.

---

#### Stratégie de Routing Groq — BASÉE SUR LES FAITS

| Scénario | Modèle | Pourquoi |
|----------|--------|----------|
| **Requête arabe** | **Qwen 3 32B** | Seul modèle avec arabe officiellement supporté + bon scores OALL + 60 RPM |
| **Requête française** | **Qwen 3 32B** | Français supporté, 119 langues, meilleur multilingue que Llama |
| **Question fiqh complexe** | **Qwen 3.6 27B** | Thinking mode + meilleur raisonnement (GPQA 87.8) |
| **Question simple/rapide** | **Qwen 3 32B** | 60 RPM = le plus rapide en disponibilité |
| **Haut volume** | **Llama 4 Scout** | 30K TPM, 500K TPD — le plus généreux en tokens |
| **Vision (image manuscrit)** | **Qwen 3.6 27B** | Seul avec vision + multilingue arabe (à vérifier sur Groq) |
| **Anti prompt-injection** | **Prompt Guard 2-86M** | Filtrage safety |

**Ce qu'on N'UTILISE PAS** :
- ❌ GPT OSS 120B/20B : faible multilingue, aucun benchmark arabe/français
- ❌ Llama 3.3 70B pour l'arabe : arabe non-supporté officiellement, rank #9 OALL
- ❌ Llama 3.1 8B pour du QA sérieux : qualité insuffisante
- ❌ Orpheus Arabic Saudi pour le Coran : pas de tajweed, dialecte ≠ coranique

### Cloud LLM — Autres Providers (Fallback)

| Provider | Modèle | Tokens gratuits/jour | Arabe | Vitesse | Meilleur pour |
|----------|--------|---------------------|-------|---------|--------------|
| OpenRouter | Qwen3-32B (free) | 20 RPM | Excellent | Modéré | Fallback si Groq down |
| Google Gemini | 2.5 Flash | ~500K-1M/jour | Bon | Rapide | Grand contexte (1M tokens), fallback |
| Cerebras | Llama 3.1 8B/70B | ~1M/jour | Moyen | 1,800 tok/s | Bulk processing |
| Mistral | Tous modèles | 1 req/s | Moyen | Rapide | Langues européennes |

### Stratégie de Routing Globale

- **Requête arabe/française** → Groq Qwen 3 32B (meilleur multilingue vérifié, 60 RPM)
- **Raisonnement complexe (fiqh/ikhtilaf)** → Groq Qwen 3.6 27B (thinking mode, GPQA 87.8)
- **Haut volume** → Groq Llama 4 Scout (30K TPM)
- **Mode offline/privacy** → Ollama Qwen2.5-7B local
- **Groq down** → OpenRouter Qwen3-32B ou Gemini
- **Contexte très long** → Gemini (1M tokens)
- **TTS interface (future)** → Groq Orpheus Arabic Saudi (hadiths/résumés SEULEMENT, PAS le Coran)
- **Input vocal (future)** → Groq Whisper Large v3
- **Anti-injection** → Prompt Guard 2-86M

### Local LLM (Fallback/Offline)

| Modèle | Params | Quantization | RAM Nécessaire | Arabe | Vitesse M4 estimée |
|--------|--------|-------------|---------------|-------|-------------------|
| Qwen2.5-7B-Instruct | 7B | Q4_K_M | ~5GB | Excellent | ~15-18 tok/s |
| Qwen2.5-14B-Instruct | 14B | Q3_K_M | ~8GB | Excellent | ~5-6 tok/s |
| Llama 3.1-8B-Instruct | 8B | Q4_K_M | ~5.5GB | Bon | ~14-16 tok/s |
| Gemma 2-9B | 9B | Q4_K_M | ~6GB | Moyen | ~8 tok/s |

**Recommandation** : Qwen2.5-7B-Instruct (Q4_K_M) — meilleur ratio qualité/taille pour le multilingue, 128K contexte, tourne confortablement sur 16GB.

**Framework** : Ollama — utilise maintenant Apple MLX backend sur Apple Silicon pour 93% de gain de vitesse.

**Optimisations Ollama** :
- OLLAMA_FLASH_ATTENTION=1 (réduit mémoire ~1GB)
- OLLAMA_KV_CACHE_TYPE=q8_0 (cache KV quantifié)
- OLLAMA_KEEP_ALIVE=30m (garde le modèle en mémoire)

---

## 8. Modèles d'Embedding

### Recommandation Principale : BAAI/bge-m3 (via FlagEmbedding)

**Pourquoi BGE-M3 a été choisi au lieu de e5-large-instruct** :
- **Hybrid natif** : dense (sémantique) + sparse (BM25-style) + ColBERT (token-level) en UN SEUL `encode()`
- **1024 dimensions**, licence MIT
- **Cross-lingual AR↔FR↔EN** validé — place français, arabe, anglais dans le même espace vectoriel
- **8192 tokens de contexte** (vs 512 pour e5-large) — essentiel pour les hadiths longs
- **FlagEmbedding.BGEM3FlagModel** requis (PAS sentence-transformers — celui-ci est dense-only)
- ~1.2 GB RAM, tourne sur Colab T4 en ~40 min pour 52K chunks
- Validé par consultation Gemini : meilleur ratio performance/couverture pour AR+FR+EN

**Pourquoi on a quitté e5-large-instruct** : e5 est excellent en dense seul, mais BGE-M3 fait dense+sparse+ColBERT dans un seul modèle. Évite de gérer un BM25 séparé. Le sparse est stocké en JSON (ChromaDB open-source ne supporte PAS les sparse vectors nativement — feature Cloud uniquement).

### Alternatives

| Modèle | Dimensions | Arabe | FR | EN | Spécialité | Licence |
|--------|-----------|-------|-----|-----|-----------|---------|
| ~~intfloat/e5-large-instruct~~ | 1024 | Bon | Bon | Bon | Dense seulement, 512 tokens | MIT |
| Omartificial/GATE-arabic-large | 1024 | **Meilleur** | Bon | Bon | SOTA arabe spécifiquement | Apache 2.0 |
| hamtaai/e5-large-hadith-v2 | 1024 | Bon | Bon | Bon | **Fine-tuné pour hadiths** | Custom |
| nomic-embed-text | 768 | Bon | Bon | Bon | Petit, via Ollama | Apache 2.0 |

**Pour le reranking** : BAAI/bge-reranker-v2-m3 — meilleur cross-encoder multilingue supportant l'arabe. 568M params, tourne sur CPU à côté d'Ollama. Améliore la précision RAG de jusqu'à 40%.

---

## 9. Vector Database

### Comparaison

| DB | Type | Coût | Meilleur pour | Hybrid Search | Metadata Filtering |
|----|------|------|--------------|--------------|-------------------|
| FAISS | Librairie | Gratuit | Simple, in-memory, rapide | Non natif | Non |
| ChromaDB | Serveur/Librairie | Gratuit | Python-native, dev facile | ⚠️ Dense seulement (sparse = Cloud uniquement) | ✅ Oui |
| Qdrant | Serveur | Gratuit (self-hosted) | Production, Rust, rapide | ✅ Natif | ✅ Avancé |
| Milvus | Serveur | Gratuit (self-hosted) | Grande échelle | ✅ Oui | ✅ Oui |
| pgvector | Extension PostgreSQL | Gratuit | Si déjà Postgres | Via extension | ✅ Via SQL |

### Recommandation
- **Développement** : ChromaDB — le plus simple en Python, metadata filtering natif
- **Production** : Qdrant — Rust-based, rapide, production-grade, hybrid search natif, gratuit self-hosted
- **Ultra-simple** : FAISS — pas de serveur, juste une librairie, parfait pour commencer

**Pour NUR** : ChromaDB pour le dense (metadata filtering essentiel pour filtrer par type/source/grade). ⚠️ **ChromaDB open-source ne supporte PAS les sparse vectors** — c'est une feature Chroma Cloud uniquement. Les sparse vectors de BGE-M3 sont stockés dans des fichiers JSON séparés (`data/sparse/{source}_sparse.json`). En Phase 2, on implémente notre propre hybrid search avec Reciprocal Rank Fusion (RRF) en combinant les scores ChromaDB (dense) et les scores sparse (JSON). On peut migrer à Qdrant plus tard si besoin.

---

## 10. Techniques RAG Avancées

### Hybrid Search (Dense + Sparse/BM25) — OBLIGATOIRE

Combine embeddings denses (similarité sémantique) avec sparse/BM25 (matching exact de termes). C'est le standard production 2025.

**Pourquoi c'est crucial pour les textes islamiques** :
- Dense search : "charité" match "zakat" conceptuellement ✅
- Sparse/BM25 : "Sahih Bukhari 1234" match exact de la référence ✅
- Ni l'un ni l'autre seul ne suffit — il faut les deux

**Implémentation NUR** :
- Dense : stocké dans ChromaDB (`{source}_dense` collections)
- Sparse : les lexical_weights de BGE-M3 stockés en JSON (`data/sparse/{source}_sparse.json`)
- Fusion : Reciprocal Rank Fusion (RRF) en Phase 2 — on combine les ranks (pas les scores bruts) des deux recherches
- ⚠️ ChromaDB open-source ne supporte PAS les sparse vectors nativement — on gère nous-mêmes

**Formule RRF** : Score(doc) = Σ 1/(k + rank_i)

**⚠️ Paramétrage critique pour texte islamique** (source : recherche Gemini 2026) :
- **k = 20-30** (PAS 60 — le standard k=60 est pour le web généraliste, trop permissif)
- **α = 0.4 dense / 0.6 sparse** — le sparse DOIT dominer car les références exactes ("البخاري 1234") et les termes de fiqh sont immuables et nécessitent un match exact
- Un k plus bas pénalise plus sévèrement les documents qui tombent bas dans le classement lexical, ce qui élimine le bruit sémantique
- Alternative : combinaison linéaire des scores normalisés (MinMax) au lieu de RRF pur : Score = α × S_dense + (1-α) × S_sparse

**Amélioration** : 15-30% de gain en recall@10 par rapport au dense seul (benchmark Atlan 2026).

### Reranking par Cross-Encoder — IMPACT MASSIF

Après le retrieval initial (top-100), un cross-encoder réévalue les paires (query, document) conjointement. Impact : jusqu'à 40% d'amélioration.

**Pipeline NUR** : Query → Dense + Sparse → RRF (top-100) → bge-reranker-v2-m3 (top-5) → LLM

**Modèle recommandé** : BAAI/bge-reranker-v2-m3 — multilingue, supporte l'arabe, 568M params, tourne sur CPU.

**⚠️ ColBERT : NE PAS utiliser au retrieval** (source : recherche Gemini 2026) :
- ColBERT (MaxSim) calcule la similarité token par token — puissant mais stockage massif + computation lourde
- Le reranker bge-reranker-v2-m3 est un cross-encoder COMPLET qui fait déjà l'attention croisée token-à-token
- Ajouter ColBERT en amont = redondant, gain MRR marginal (<2%), complexité disproportionnée
- **Conclusion** : Dense + Sparse + Reranker (sans ColBERT) = meilleur ratio coût/précision

### Contextual Retrieval (Anthropic) — OBLIGATOIRE POUR NUR

Préfixer chaque chunk avec du contexte LLM-généré. -67% d'échecs de retrieval.

**Pourquoi c'est critique pour les textes islamiques** (source : recherche Gemini 2026) :
- Les versets du Coran dépendent massivement du contexte de révélation (Asbab al-Nuzul)
- Un verset isolé peut perdre son cadre d'application (ex: "Et tuez-les..." sans le contexte de légitime défense)
- Les hadiths utilisent des pronoms ("Il", "Ils") sans antécédent clair hors contexte

**Compatibilité BGE-M3** : ✅ Parfaite — BGE-M3 a 8192 tokens de contexte, ajouter 50-100 mots de contexte ne sature rien

**⚠️ Piège critique** : Le LLM qui génère le contexte enrichi NE DOIT PAS ajouter d'interprétation dogmatique. Prompt purement factuel : nom du livre, chapitre, sujet global, locuteur.

**Coût** : 52K chunks est un volume faible — quelques heures de ré-embedding pour un gain de sécurité théologique drastique. Ça vaut 100% le coût.

**Exemple** :
- Avant : `"Et tuez-les où que vous les rencontriez..."`
- Après : `"[Ce chunk est extrait de la Sourate Al-Baqarah, verset 191, traitant des règles de légitime défense lors des combats contre les persécuteurs de la Mecque]. Et tuez-les où que vous les rencontriez..."`

### Parent-Child Chunking

**Principe** :
- Child chunks (100-200 tokens) — pour le retrieval haute précision
- Parent chunks (500-1000 tokens) — envoyés au LLM pour contexte complet
- Quand assez de child chunks du même parent sont retrieved, on "remonte" au parent

**Application islamique** :
- Retrieve sur un ayah spécifique (child) → fournir toute la section thématique (parent)
- Retrieve sur un hadith spécifique (child) → fournir tout le chapitre (parent)

### Corrective RAG (CRAG)

Ajoute un évaluateur de retrieval qui grade les documents AVANT de les passer au générateur :
- Confiance haute → procéder avec RAG normal
- Confiance moyenne → reformuler la requête et re-retrouver
- Confiance basse → dire "Je n'ai pas suffisamment d'information" ou fallback web

**Critique pour les textes religieux** : En islam, une mauvaise réponse est pire que pas de réponse. CRAG garantit qu'on ne génère que quand on a des sources fiables.

### Adaptive RAG

Route dynamiquement les requêtes vers la stratégie de retrieval appropriée :
- Question factuelle simple → Recherche vectorielle directe
- Question comparative fiqh → Multi-query + Graph RAG
- Question générale → Pas de retrieval (LLM direct)
- Question hors domaine → Refuser poliment

### Graph RAG

Utilise un graphe de connaissances pour capturer les relations entre concepts islamiques. Particulièrement puissant parce que :
- Les chaînes de narration (isnad) sont des structures de graphe naturelles
- Les concepts coraniques ont des interconnexions profondes (zakat → charité → purification → justice sociale)
- Coran → Tafsir → Fiqh → Fatwa est une chaîne multi-hop

**Ontologie coranique existante** : Quranic Arabic Corpus (corpus.quran.com) — 350 concepts (285 collectés), structurés en arbre. Source pour le Graph RAG.

### Agentic RAG

Enveloppe le pipeline RAG dans un agent qui peut utiliser des outils, raisonner sur le retrieval, et prendre des actions multi-étapes :
1. Classifier la question (fiqh, aqeedah, seerah, etc.)
2. Router vers le corpus approprié (Coran, hadith, tafsir, savants)
3. Retrieving et vérifier les sources
4. Cross-référencer entre sources multiples
5. Noter l'ikhtilaf quand il existe
6. Générer une réponse fondée et citée

---

## 11. Anti-Hallucination — Pipeline de Vérification

### Les 6 Types d'Hallucination RAG (Taxonomie arXiv 2601.19927, Jan 2026)

1. **Hallucination par surconfiance** — Le modèle affirme avec confiance quelque chose qui n'est pas dans les sources
2. **Hallucination par obsolescence** — Sources récupérées contenant des informations supplantées (fatwas dépassées)
3. **Hallucination par invérifiabilité** — Affirmations qu'on ne peut retracer à aucune source récupérée
4. **Déviation d'instruction** — La réponse ne traite pas la question réelle
5. **Inconsistence de contexte** — Les contextes récupérés se contredisent (fréquent en islam avec les multiples opinions)
6. **Déficience de raisonnement** — Le modèle fait des sauts logiques non supportés par les preuves

### Pipeline de Mitigation en 6 Modules (Mis à jour — recherche Gemini 2026)

**Module T1 — Query Refining (Pré-retrieval)** :
- Reformulation de la requête pour plus de clarté
- Multi-query : générer 3-5 variantes pour améliorer le recall
- Step-back prompting : générer une question plus générale pour contexte plus large
- **Détection de hadith cité par l'utilisateur** : si la question contient une citation arabe entre guillemets, l'isoler pour vérification dans la base des hadiths fabriqués

**Module T2 — Reference Identification (Post-retrieval)** :
- Reranking pour améliorer la pertinence
- Filtrage par pertinence (CRAG)
- Vérification du grade d'authenticité
- **Divergence Detection** : si le reranker renvoie des chunks aux conclusions opposées avec des scores similaires → lever un flag `Divergence=True` qui force l'affichage multi-perspectives

**Module T3 — Prompt Engineering (Pré-generation)** :
- **In-Context Isolation** : interdiction FORMELLE d'utiliser les connaissances pré-entraînées du LLM
- **Chain-of-Thought Citation** : le LLM doit D'ABORD lister les IDs des sources valides, ENSUITE rédiger
- Format XML structuré pour les chunks dans le prompt (voir Section Prompt Engineering détaillée)
- Source ID injection (Pilier 7)
- Instructions de distinction Coran/Hadith/Savant
- Instruction d'abstention si incertain
- **Structured Output (JSON Schema via Groq)** : forcer le LLM à produire `{exact_citations: [...], synthesis: "..."}` — si un mot dans exact_citations n'est pas dans le contexte, rejet au niveau du parser

**Module T4 — NLI Verification (Post-génération, NOUVEAU)** :
- **Modèle NLI** (DeBERTa-v3-large ou Ayn-NLI arabe) vérifie si chaque phrase de la réponse est strictement supportée par les chunks sources
- Score d'entailment < 0.95 → la phrase est rejetée
- C'est la couche de protection la plus importante — indépendante du LLM qui a généré la réponse

**Module T5 — Caractère par Caractère (Post-génération)** :
- **Bi-level Character Matching** pour l'arabe :
  - Niveau 1 (Sémantique) : Smith-Waterman ou Levenshtein après strip diacritiques + normalisation
  - Niveau 2 (Strict) : comparaison de la sous-chaîne isolée avec la source exacte (Coran de Médine/Uthmani) — match 100% sur les lettres de base
- Vérification des références hadith : collection + numéro existe-t-il ?
- Score de fidélité par affirmation
- Remplacement des citations fausses ou avertissement

**Module T6 — Decoupled Grounding (Structuré, NOUVEAU)** :
- Le LLM doit générer la réponse en DEUX blocs séparés via Structured Output (JSON Schema) :
  1. `exact_citations` : citations mot-à-mot extraites du contexte
  2. `synthesis` : texte de liaison qui relie les citations
- Si un mot dans `exact_citations` n'est pas présent à 100% dans le contexte → rejet au niveau du parser backend (avant même d'afficher à l'utilisateur)
- **Constrained Decoding** (Outlines / Instructor) : on peut forcer le LLM à ne produire que des tokens présents dans les chunks récupérés pour les champs de citation arabe

### Citation Grounding

**Problème** : Le LLM peut citer des sources qui n'existent pas ou mal attribuer du contenu.

**Solution** :
- Seules les citations du set de documents récupérés sont permises
- Après génération, vérifier que chaque citation mappe à un chunk réel
- Si une citation ne mappe à rien → la supprimer et avertir

### Détection de Hadiths Fabriqués (Mawdu') — Stratégie Double-RAG

**Problème** : Si un utilisateur cite un hadith fabriqué qui n'est pas dans nos 33,738 hadiths authentiques.

**Solution en 2 étapes** (source : recherche Gemini 2026) :
1. **Vérification lexicale stricte** : Quand l'utilisateur cite un texte arabe, isoler la citation et chercher dans la base authentique (sparse/BM25)
2. **Base de données négative** : Indexer les hadiths fabriqués connus (dataset **ADAM-HA** — 100K+ entrées labellisées authentiques vs fabriquées, + ouvrages d'Al-Albani sur les hadiths faibles)
3. **Stratégie** :
   - Hadith trouvé dans base authentique → réponse normale
   - Hadith trouvé dans base fabriquée → avertir : "Ce hadith est classé comme Mawdu'/Faible par [Savant]"
   - Hadith trouvé nulle part (cosine < 0.45) → "Ce texte ne figure pas dans les corpus authentiques de référence indexés. Par précaution religieuse, NUR ne peut confirmer sa véracité."

### Ikhtilaf — Framework de Neutralité

**Problème** : Comment présenter les désaccords savants sans prendre position.

**Solution** (source : recherche Gemini 2026) :
- **Metadata tagging obligatoire** lors de l'ingestion : `madhab: [hanafi|maliki|shafi'i|hanbali]`, `savant: [nom]`
- **Divergence Detection** : si le reranker renvoie des chunks aux conclusions opposées avec des scores similaires → flag `Divergence=True`
- **Prompt d'arbitrage neutre** : "Il existe une divergence d'opinion (Ikhtilaf) légitime sur cette question parmi les grands savants. Voici les avis documentés : Avis A (École X) : [Preuve]. Avis B (École Y) : [Preuve]."
- **Interdiction stricte** : le LLM ne JAMAIS déclarer un avis comme "correct" ou "supérieur" de son propre chef

---

## 12. Sources Clickables — URLs

Chaque source citée doit avoir un lien clickable direct vers le texte original en ligne. Les URLs sont prévisibles et stables.

### URLs par Type de Source

**Coran** : `https://quran.com/{surah}/{ayah}`
- Exemple : https://quran.com/2/255 → Al-Baqarah, Ayah 255

**Hadiths** : `https://sunnah.com/{collection}:{hadith_number}`
- Sahih Bukhari : `https://sunnah.com/bukhari:{number}`
- Sahih Muslim : `https://sunnah.com/muslim:{number}`
- Sunan Abu Dawud : `https://sunnah.com/abudawud:{number}`
- Jami at-Tirmidhi : `https://sunnah.com/tirmidhi:{number}`
- Sunan an-Nasai : `https://sunnah.com/nasai:{number}`
- Sunan Ibn Majah : `https://sunnah.com/ibnmajah:{number}`
- Exemple : https://sunnah.com/bukhari:1 → Premier hadith de Bukhari

**Tafsir** : `https://quran.com/tafsir/{surah}:{ayah}`
- Exemple : https://quran.com/tafsir/2/255 → Tafsir d'Al-Baqarah 255

**Fatwas IslamQA** : `https://islamqa.info/{language}/answers/{fatwa_id}`
- Anglais : `https://islamqa.info/en/answers/{id}`
- Français : `https://islamqa.info/fr/answers/{id}`
- Arabe : `https://islamqa.info/ar/answers/{id}`

**Livres classiques** : `https://shamela.ws/book/{book_id}`

### Dans l'Interface

Chaque citation est renderisée avec un lien :
- 📖 **Al-Baqarah 2:255** → lien vers quran.com
- 📚 **Sahih Bukhari #1234** → lien vers sunnah.com
- 📝 **Tafsir Ibn Kathir — 2:255** → lien vers quran.com/tafsir
- 🎓 **IslamQA #9934** → lien vers islamqa.info

---

## 13. Avis des Savants — Obligation Éthique

### Pourquoi c'est non-négociable

En islam, l'interprétation des textes sacrés (Coran et Hadith) n'est pas ouverte aux non-savants. C'est un principe fondamental : **le tafsir (exégèse) et le istinbat (déduction de rulings) nécessitent des compétences spécifiques**.

Un layman qui interprète seul risque :
- **Mauvaise compréhension** du contexte de révélation (asbab al-nuzul)
- **Commission de haram** par ignorance des exceptions et conditions
- **Bid'ah** (innovation religieuse) en ajoutant des pratiques non fondées
- **Shirk** (associationisme) dans le pire cas, par mauvaise interprétation d'attributs divins

### Ce que NUR fait

- Le système ne DONNE JAMAIS son propre avis
- Le système rapporte TOUJOURS les avis de savants reconnus avec leurs sources
- Si un avis est un ijma' (consensus) → le mentionner
- S'il y a ikhtilaf (désaccord) → présenter tous les avis (voir Pilier 9)
- Toujours ajouter un disclaimer : "Pour une application spécifique, consultez un savant qualifié"

### Sources de données savantes (toutes gratuites)

**Fatwas et opinions institutionnelles** :
- IslamQA.info — supervision du Sheikh Abdurrahman ibn Salih al-Mahmud, savants salafis
- Islamweb.net — Centre du Qatar pour la fatwa, savants reconnus
- Dar al-Ifta al-Misriyyah — institution officielle égyptienne, Grand Mufti

**Livres classiques** (via Shamela.ws et autres) :
- Al-Mughni (Ibn Qudamah) — Hanbali fiqh
- Al-Muhadhdhab (Shirazi) — Shafi'i fiqh
- Al-Mabsout (Sarakhsi) — Hanafi fiqh
- Al-Mudawwana (Sahnun) — Maliki fiqh
- Fiqh us-Sunnah (Sayyid Sabiq) — synthèse comparative
- Al-Muwatta (Imam Malik) — Hadith + Fiqh
- Sahih Bukhari/Muslim avec commentaires (Ibn Hajar, Nawawi)

**Savants contemporains reconnus** :
- Ibn Baz, Ibn Uthaymeen, Al-Albani — tradition salafi
- Yusuf al-Qaradawi — tradition réformiste
- Ali Gomaa — tradition ash'arite/Grand Mufti Egypte

---

## 14. Ikhtilaf Awareness — Conscience du Désaccord Savant

### Qu'est-ce que l'ikhtilaf

L'ikhtilaf (اختلاف) est le désaccord savant sur une question de fiqh. C'est un phénomène reconnu et respecté dans la tradition islamique. L'Imam Shafi'i a dit : "Mon avis est correct avec la possibilité d'erreur, et l'avis des autres est en erreur avec la possibilité d'être correct."

### Pourquoi c'est crucial

- Présenter un seul avis comme vérité = théologiquement incorrect
- Peut amener l'utilisateur à croire qu'un acte est haram alors que d'autres savants le permettent (ou inversement)
- Peut créer du sectarisme en privilégiant une école sans mentionner les autres
- La tradition islamique respecte la diversité d'opinion dans le fiqh

### Les niveaux de consensus — PRÉSENTATION OBLIGATOIRE

L'ikhtilaf n'est pas binaire (accord vs désaccord). Il existe des **degrés de consensus** que le système DOIT refléter clairement :

**Niveau 1 — Ijma' (Consensus unanime)** :
Tous les savants de toutes les écoles sont d'accord. C'est le niveau le plus fort. Le système affiche : "Il y a consensus (ijma') entre tous les savants sur ce point."

**Niveau 2 — Majorité écrasante** :
La quasi-totalité des écoles/savants est d'accord, avec un avis dissident très minoritaire. Le système affiche clairement : "La majorité des savants [X, Y, Z] sont d'accord sur [avis]. L'école [A] a un avis différent : [avis minoritaire]."

**Niveau 3 — Désaccord significatif** :
Plusieurs écoles ont des avis différents, sans majorité claire. Le système affiche chaque avis avec ses preuves, sans hiérarchiser.

**Niveau 4 — Avis isolé (1 école contre toutes les autres)** :
C'est le cas le plus délicat et le plus dangereux. Si 4 écoles sur 4 (ou 4 sur 5) rejettent un avis et qu'une seule l'accepte, le système DOIT :
- Mentionner clairement que c'est un avis isolé/minoritaire
- Indiquer combien d'écoles sont POUR et CONTRE
- Présenter les preuves de chaque côté
- **NE JAMAIS** ajouter un message du type "C'est parce qu'il y a une école qui l'accepte que c'est permis" ou "Tu peux suivre cet avis" — cela inciterait l'utilisateur à suivre l'avis minoritaire
- **NE JAMAIS** non plus ajouter un message du type "Cet avis est rejeté par la majorité, ne le suis pas" — cela pourrait être perçu comme imposant une école sur une autre
- **Présenter les faits neutrement** : quelles écoles disent quoi, avec leurs preuves, et laisser le musulman choisir à ses propres risques et périls

### Le principe de neutralité absolue

**La crainte fondatrice** : Le système ne doit JAMAIS influencer l'utilisateur vers un avis ou un autre. Ni encourager le suivi d'un avis minoritaire ("c'est permis car une école l'accepte"), ni décourager ("ne suis pas cet avis car la majorité est contre"). Le musulman est responsable de ses propres choix devant Allah. Le rôle du système est d'informer, pas de guider.

**Règle** : Le système présente les faits — qui dit quoi, pourquoi, sur quelle preuve — et s'arrête là. Le choix appartient au musulman, à ses propres risques et périls, idéalement en consultant un savant qualifié.

### Comment NUR gère l'ikhtilaf

**Cas 1 — Consensus (Ijma')** :
Si tous les savants sont d'accord → réponse simple avec mention du consensus
Exemple : "Les 5 prières quotidiennes sont obligatoires — il y a consensus (ijma') sur ce point."

**Cas 2 — Majorité écrasante avec avis minoritaire** :
Afficher : "La majorité des savants [écoles X, Y, Z] soutiennent [avis A] basé sur [preuves]. L'école [W] a un avis différent : [avis B] basé sur [preuves]."
Sans juger, sans encourager, sans décourager.

**Cas 3 — Désaccord significatif (ikhtilaf classique)** :
Présenter CHAQUE avis avec :
- L'école/savant qui le soutient
- La preuve (Coran/Hadith) sur laquelle il se base
- Le raisonnement si disponible
- Ne JAMAIS dire quel avis est "meilleur" ou "plus correct"
- Indiquer clairement le nombre d'écoles pour vs contre

**Cas 4 — Avis isolé (1 école contre toutes les autres)** :
- Mentionner EXPLICITEMENT : "Sur cette question, [X] écoles sur [Y] soutiennent [avis A]. Seule l'école [Z] soutient [avis B]."
- Présenter les preuves des DEUX côtés
- Ne RIEN ajouter qui encourage ou décourage le suivi de l'avis minoritaire
- Laisser le musulman décider à ses propres risques et périls

**Cas 5 — Questions de aqeedah (croyance)** :
Pas d'ikhtilaf toléré — la croyance est basée sur les textes authentiques. Si un avis contredit le Coran ou les hadiths sahih, il n'est pas présenté comme valide.

**Disclaimer systématique** :
"Toute question de fiqh peut avoir des avis différents selon l'école de jurisprudence. Pour une application à votre situation personnelle, consultez un savant qualifié de votre école."

---

## 15. Bilingual Strategy — Arabic-First

### Le Principe Fondamental

**Le texte arabe est la source de vérité.** Les traductions sont de l'aide à la compréhension, pas des sources primaires.

### Pourquoi les traductions sont problématiques

- Le Coran en arabe a des nuances qu'aucune traduction ne peut capturer
- Un mot arabe peut avoir 10+ sens selon le contexte
- Les traducteurs font des choix d'interprétation qui peuvent être discutables
- La parole d'Allah EST en arabe — la traduction est une approximation humaine

### L'approche Arabic-First Cross-Lingual Retrieval

1. **Indexation** : Le texte arabe est le contenu principal embeddé dans la base vectorielle
2. **Recherche** : Le modèle d'embedding multilingue (BGE-M3) matche DIRECTEMENT la question française/anglaise contre les chunks arabes — pas besoin de traduire la requête
3. **Affichage** : Montrer le texte arabe original + traduction française pour compréhension
4. **Vérification** : Toujours vérifier contre le texte arabe, jamais contre la traduction

### Pourquoi ça marche sans traduction

Les modèles comme multilingual-e5-large-instruct sont entraînés sur des milliards de paires multilingues. Ils "comprennent" que "zakat" (FR), "zakat" (EN), et "زكاة" (AR) réfèrent au même concept. L'espace vectoriel est partagé entre les langues. Le matching cross-lingual est natif.

### Bénéfices

- Zéro latence de traduction
- Zéro perte de précision due à la traduction
- Le texte sacré arabe est toujours la source de vérité
- Fonctionne pour toute langue supportée par le modèle d'embedding

---

## 16. Débat Framework — LangChain vs Custom

### Position Finale : Hybride Intelligent

**LangGraph (Orchestration)** : Utilisé pour le StateGraph du pipeline RAG complet. Routing adaptatif (Coran? Hadith? Fiqh?), CRAG (évaluation du retrieval), mémoire de conversation, streaming. Le faire custom = des centaines de lignes pour réinventer ce que LangGraph fait nativement.

**Custom Python (Islamic Core)** : Toute la logique islamique que personne d'autre n'a. Normalisation arabe, scoring par authenticité, moteur de vérification de citations, détection et formatage de l'ikhtilaf, génération de liens sources, cross-reference mapping, extraction d'avis savants, vérification caractère par caractère du Coran. AUCUN framework ne sait faire ça.

**LangChain (Intégrations standard)** : Pour le plumbing — connecter Ollama, Groq, FAISS/ChromaDB, BM25 hybrid search, Pydantic output parsing. Réinventer ces intégrations = perte de temps.

### Arguments Pour LangChain

- Standard de l'industrie — n'importe quel dev Python connaît
- Intégrations prêtes pour Ollama, Groq, FAISS, ChromaDB, BM25
- LangGraph pour l'orchestration agentic (CRAG, routing) ultra-propre
- Conversation memory intégré
- PydanticOutputParser pour forcer le format de sortie du LLM
- Streaming natif token par token
- Communauté massive + documentation

### Arguments Contre LangChain

- Poids : langchain + langchain-core + langchain-community = ~150MB
- Abstractions opaques — 5 couches d'abstraction rendent le debug difficile
- Breaking changes fréquents entre versions
- Overhead de latence sur chaque RunnableLambda
- Rien pour le domaine islamique — on doit tout custom de toute façon

### Pourquoi le pur custom n'est pas optimal

Réinventer les intégrations Ollama/Groq/FAISS, la conversation memory, le streaming, le routing adaptatif = des centaines d'heures pour des problèmes déjà résolus. Mieux vaut utiliser ce qui existe et investir le temps dans ce qui n'existe pas (la logique islamique).

---

## 17. Contraintes Hardware — MacBook M4 16GB

### Budget Mémoire

| Composant | Mémoire Estimée | Notes |
|-----------|----------------|-------|
| macOS | ~3-4 GB | Non-négociable |
| Ollama (7B Q4) | ~4-5 GB | Qwen2.5-7B Q4_K_M |
| Modèle embedding | ~0.5-1 GB | sentence-transformers |
| Reranker (bge-m3) | ~1-1.5 GB | Tourne sur CPU |
| ChromaDB/FAISS | ~0.5-1 GB | Index + cache |
| FastAPI + Python | ~0.5-1 GB | Runtime overhead |
| **Total** | **~10-13 GB** | **Serré mais faisable** |

### Stratégies d'Optimisation

- **Sequential model loading** : Ollama peut unload/load les modèles à la demande
- **Embedding via Ollama** : nomic-embed-text via Ollama (~274MB) au lieu d'un process Python séparé → réduit la mémoire totale
- **Reranker sur CPU** : libère la mémoire GPU pour Ollama
- **ChromaDB in-memory** : pour <100K chunks, pas besoin de serveur séparé
- **Sémaphore de concurrence** : limiter à 3 requêtes LLM simultanées
- **Single worker uvicorn** : le modèle Ollama est chargé une fois en mémoire

### Ordinateurs compatibles

L'architecture est conçue pour tourner sur MacBook Air M4 16GB. Tous les composants sont dimensionnés pour cette contrainte. Sur une machine plus puissante, on peut monter en modèle (Qwen2.5-14B au lieu de 7B, plus de données indexées, etc.).

---

## 18. Évaluation & Métriques

### RAGAS — Métriques Standard

| Métrique | Ce qu'elle mesure | Cible |
|----------|------------------|-------|
| Faithfulness | La réponse est-elle fondée dans le contexte récupéré ? | > 0.90 |
| Answer Relevancy | La réponse répond-elle à la question ? | > 0.85 |
| Context Precision | Les chunks pertinents sont-ils mieux classés ? | > 0.80 |
| Context Recall | Tous les chunks nécessaires sont-ils récupérés ? | > 0.85 |

### Métriques Spécifiques Islamique

| Métrique | Ce qu'elle mesure |
|----------|------------------|
| Source Attribution Accuracy | Chaque affirmation factuelle cite-t-elle correctement sa source ? |
| Madhhab Consistency | Le système maintient-il la cohérence avec l'école référencée ? |
| Ikhtilaf Awareness | Le système note-t-il le désaccord savant quand il existe ? |
| Quranic Accuracy | Les versets coraniques sont-ils cités verbatim (pas paraphrasés) ? Les numéros d'ayah sont-ils corrects ? |
| Hadith Grading | Le système note-t-il le grade d'authenticité ? |
| Abstention Rate | Le système refuse-t-il correctement les questions hors scope ? |
| Cross-reference Consistency | Quand le même thème apparaît dans Coran, hadith, et fiqh, le système synthétise-t-il correctement ? |

### Le RAG Triad (Diagnostic Simplifié)

1. **Context Relevance** → Est-ce que le retriever trouve les bons documents ?
2. **Faithfulness** → Est-ce que le générateur s'en tient au contexte ?
3. **Answer Relevance** → La réponse répond-elle à la question ?

**Diagnostic** :
- Low context relevance + High faithfulness = Bon générateur, mauvais retriever → Fixer le retrieval
- High context relevance + Low faithfulness = Bon retriever, mauvais générateur → Fixer le prompt/modèle
- Low answer relevance = Problème de compréhension de la question → Fixer le query processing

---

## 19. Recherche Académique — Papiers Clés

| Papier | Référence | Contribution |
|--------|-----------|-------------|
| Islamic Chatbots in the Age of LLMs | arXiv: 2601.06092 | Analyse complète des chatbots islamiques LLM-powered |
| RAG in Quranic Studies | arXiv: 2503.16581 | RAG améliore significativement la qualité des réponses pour QA coranique |
| Improving LLM Reliability with RAG in Religious QA | arXiv: 2401.15378 | Propose système RAG pour minimiser hallucinations en questions religieuses |
| Aqrag: Advanced Quranic RAG | Springer 2026 | QRCD v1.2 dataset, Advanced Quranic RAG |
| Faithful & Advanced RAG for Islamic QA | arXiv: 2510.25621 | Architecture RAG itérative et adaptative pour QA islamique persan |
| Agentic Quran-Grounding Framework | emergentmind.com | RAG basé sur des outils assurant fidélité et traçabilité |
| Implementing a Sharia Chatbot | arXiv: 2512.16644 | Chatbot sharia-compliant pour questions islamiques |
| Attribution Techniques for Mitigating Hallucinated Information in RAG | arXiv: 2601.19927 (Jan 2026) | Taxonomie des 6 types d'hallucination RAG, pipeline de mitigation en 4 modules |
| Swan and ArabicMTEB | arXiv: 2411.01192 | Benchmark complet pour modèles d'embedding arabe |
| GATE Arabic Embedding | arXiv: 2505.24581 | SOTA pour embedding arabe |

---

## 20. Projets Open-Source — Analyse Détaillée

### Résumé de Tous les Projets Analysés

| Projet | Architecture | LLM | Embedding | Vector DB | Grading Hadith | Citation | Arabe | Français |
|--------|-------------|-----|-----------|-----------|---------------|----------|-------|---------|
| Hadith-AI | LlamaIndex RAG | Qwen2.5 7B | qwen3-embedding 4B | Qdrant | ❌ Ignoré | ❌ Basique | ✅ | ❌ |
| Islam-GPT | LangChain RAG | Mistral 7B | MiniLM-L6-v2 | FAISS | ❌ Absent | ❌ Raw dump | ❌ | ❌ |
| AraQA | Generate-then-verify | AraGPT2 135M | multilingual-e5-base | Aucun | ❌ | ⚠️ Fragile | ✅ | ❌ |
| Quran RAG | Custom RAG | Llama 3.2 1B | MiniLM-L6-v2 | FAISS | N/A | ❌ | ❌ | ❌ |
| HadithRAG | LangChain RAG | GPT-3.5 | OpenAI ada | ChromaDB | ❌ Ignoré | ⚠️ Partiel | Metadata | ❌ |
| Quran-Hadith | LangChain RAG | Gemini Pro | MiniLM-L6-v2 | ChromaDB | ❌ | ❌ | ❌ | ❌ |
| IslamAI (oshoura) | LangChain RAG | GPT-3.5 | OpenAI ada | Pinecone | ❌ | ✅ Accordion | ❌ | ❌ |
| IslamAI (yousef) | Flask API | Aucun | Aucun | Aucun | N/A | N/A | ✅ | ❌ |
| Chatbot-QnA | Fine-tune | T5/BART/Pegasus | N/A | Milvus (planifié) | N/A | ❌ | ❌ | ❌ |
| Fin-Islam | Multi-Agent RAG | Llama 3.1 8B | nomic-embed-text | pgvector | N/A | ✅ | ❌ | ❌ |
| Graph RAG | Dual Semantic+Graph | GPT-4o-mini | text-embedding-3-small | Précalculé | N/A | ❌ | ❌ | ❌ |

### Innovations uniques par projet

- **Hadith-AI** : SentenceWindowNodeParser pour contexte autour des chunks
- **Islam-GPT** : Rien d'unique (fork de chatbot médical)
- **AraQA** : Seul avec vérification post-génération (regex + cosine similarity)
- **Quran RAG** : Rien d'unique
- **HadithRAG** : Corpus le plus riche (50K+ hadiths, 30+ métadonnées)
- **Quran-Hadith** : Query expansion via LLM avant retrieval
- **IslamAI (oshoura)** : Meilleur anti-hallucination prompt, domain-aware question condensing, disclaimer fatwa
- **IslamAI (yousef)** : Collection de données la plus complète (400+ fichiers)
- **Chatbot-QnA** : Chain-of-Thought pour raisonnement islamique
- **Fin-Islam** : Compliance Agent (seul avec validation post-génération Sharia), MCP, streaming SSE
- **Graph RAG** : Dual semantic+graph retrieval, verse embedding enhancement

### Ce qu'on emprunte (avec amélioration)

- **De AraQA** : Le concept de vérification post-génération → mais avec méthodes robustes au lieu de regex
- **De HadithRAG** : Le schéma de métadonnées riche → mais avec arabe comme source primaire
- **De IslamAI (oshoura)** : Le prompt anti-hallucination et le disclaimer fatwa → mais étendu avec avis savants
- **De Quran-Hadith** : Le query expansion → mais combiné avec détection de langue
- **De Fin-Islam** : Le Compliance Agent → adapté en "Islamic Guardrail Agent"
- **De Graph RAG** : L'ontologie coranique et le verse embedding enhancement → intégré en Phase 4

---

## 21. Matrice Comparative Finale

| Feature | Daleel | Hadith-AI | Islam-GPT | HadithRAG | fin-islam | IslamAI oshoura | Graph RAG | **NUR** |
|---------|--------|-----------|-----------|-----------|-----------|----------------|-----------|---------|
| Triple Index | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 4 indexes |
| Arabic-First | ❓ | ⚠️ Partiel | ❌ | ⚠️ Metadata | ❌ | ❌ | ❌ | ✅ Cross-lingual |
| Auth Weighting | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Grade boost |
| Post-Gen Verification | ❓ | ❌ | ❌ | ❌ | ⚠️ Compliance | ❌ | ❌ | ✅ Char-by-char |
| Context-Enriched | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Verse embed | ✅ Anthropic technique |
| Structured Citations | ✅ | ❌ | ❌ | ⚠️ | ✅ | ✅ | ❌ | ✅ Source ID Protocol |
| Ikhtilaf Awareness | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Multi-madhhab |
| Scholar Opinions | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Avec sources |
| Hybrid Search | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Dense + BM25 |
| Reranking | ❓ | ⚠️ Mauvais | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ bge-reranker-v2-m3 |
| Clickable Sources | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ URLs stables |
| French Support | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Natif |
| 100% Gratuit | ❌ Premium | ✅ | ✅ | ⚠️ OpenAI | ⚠️ Ollama | ❌ OpenAI | ❌ OpenAI | ✅ |
| 100% Local Option | ❌ | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| Quran Verification | ❓ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Char-by-char |

---

## 22. Appendice — Toutes les Références

### APIs de Données

- Al Quran Cloud API : https://alquran.cloud/api
- fawazahmed0 Quran API : https://github.com/fawazahmed0/quran-api
- fawazahmed0 Hadith API : https://github.com/fawazahmed0/hadith-api
- Quran.com API : https://quran.com/en/developers
- QUL Tarteel : https://qul.tarteel.ai
- Sunnah.com API : https://sunnah.com/developers
- Sunnah.now API : https://sunnah.now / https://docs.sunnah.now
- Tafsir API : https://github.com/spa5k/tafsir_api
- HadithAPI.com : https://www.hadithapi.com
- The9Books API : https://github.com/mghanii/The9Books
- HadeethEnc API : https://documenter.getpostman.com/view/5211979/TVev3j7q

### Datasets

- AhmedBaset Hadith JSON : https://github.com/AhmedBaset/hadith-json
- Hadith HuggingFace : https://huggingface.co/datasets/fawazahmed0/hadith-data
- Hadith Kaggle : https://www.kaggle.com/datasets/fahd09/hadith-dataset
- Multilingual Quran HuggingFace : https://huggingface.co/datasets/anisafifi/multilingual-quran
- Sunnah Dataset HuggingFace : https://huggingface.co/datasets/meeAtif/hadith_datasets
- Tafsir Ibn Kathir Kaggle : https://www.kaggle.com/datasets/oyilmaztekin/quran-tafsir-ibn-kathir-jsonl
- Quran Audio HuggingFace : https://huggingface.co/datasets/tarteel-ai/everyayah

### Projets Open-Source

- Hadith-AI : https://github.com/ahmedeltaher/Hadith-AI
- HadithRAG : https://github.com/Quchluk/HadithRAG
- Quran RAG : https://github.com/Shariar-Mozumder/Quran_RAG
- Islam-GPT : https://github.com/Shaheer66/Islam-GPT
- IslamAI (yousef) : https://github.com/yousefabuz17/IslamAI
- IslamAI (oshoura) : https://github.com/oshoura/IslamAI
- Quran-Hadith-Chatbot : https://github.com/hammadali1805/Quran-Hadith-Chatbot
- Chatbot-QnA-Quran : https://github.com/Bayhaqieee/Chatbot-QnA-Quran
- Fin-Islam : https://github.com/dannycahyo/fin-islam
- AraQA : https://github.com/Marje3na/AraQA-An-Arabic-Generative-Question-Answering-Model-for-Authentic-Religious-Text
- Arabic RAG Pipeline : https://github.com/zenmakhlouf/arabic-rag-pipeline
- Awesome Islamic Apps : https://github.com/tarekeldeeb/awesome-islamic-open-source-apps
- Islamic Agent (Graph RAG) : https://github.com/seelenbrecher/islamic-agent
- Hadith Hub : https://hadithhub.com

### Modèles d'Embedding

- multilingual-e5-large-instruct : https://huggingface.co/intfloat/multilingual-e5-large-instruct
- bge-m3 : https://huggingface.co/BAAI/bge-m3
- bge-reranker-v2-m3 : https://huggingface.co/BAAI/bge-reranker-v2-m3
- GATE Arabic : https://huggingface.co/collections/Omartificial-Intelligence-Space/arabic-matryoshka-and-gate-embedding-models
- Hadith-specific embedding : https://huggingface.co/hamtaai/e5-large-hadith-v2
- nomic-embed-text (Ollama) : https://ollama.com/library/nomic-embed-text
- Jina-ColBERT-v2 : https://jina.ai/news/jina-colbert-v2

### Providers LLM

- Ollama : https://ollama.com
- LM Studio : https://lmstudio.ai
- Google Gemini API : https://ai.google.dev
- Groq : https://console.groq.com
- OpenRouter (free models) : https://openrouter.ai/collections/free-models
- Cerebras : https://www.cerebras.ai/inference
- Mistral : https://docs.mistral.ai
- Together AI : https://www.together.ai
- Free LLM API Resources : https://github.com/cheahjs/free-llm-api-resources

### Papiers Académiques

- Islamic Chatbots in Age of LLMs : https://arxiv.org/abs/2601.06092
- RAG in Quranic Studies : https://arxiv.org/html/2503.16581v1
- Improving LLM Reliability with RAG in Religious QA : https://arxiv.org/pdf/2401.15378
- GATE Arabic Embedding : https://arxiv.org/html/2505.24581v1
- Swan & ArabicMTEB : https://arxiv.org/html/2411.01192v2
- Aqrag Quranic RAG : https://link.springer.com/10.1007/s00521-026-11886-7
- Sharia Chatbot : https://arxiv.org/html/2512.16644
- RAG Hallucination Taxonomy : arXiv 2601.19927 (Jan 2026)
- Trustworthy Islamic Agent (Graph RAG) : https://medium.com/@maulanaanab/trustworthy-islamic-agent-a-graph-based-rag-for-islamic-content-using-a-quran-ontology-be7e1d729315
- Anthropic Contextual Retrieval : https://www.anthropic.com/engineering/contextual-retrieval

### Frameworks RAG

- LangChain : https://python.langchain.com
- LangGraph : https://langchain-ai.github.io/langgraph
- LlamaIndex : https://docs.llamaindex.ai
- RAGAS : https://docs.ragas.io
- DeepEval : https://github.com/confident-ai/deepeval

### Sources Clickables

- Quran.com : https://quran.com
- Sunnah.com : https://sunnah.com
- IslamQA.info : https://islamqa.info
- Islamweb.net : https://islamweb.net
- Shamela.ws : https://shamela.ws
- Quranic Arabic Corpus : https://corpus.quran.com

### Outils de Développement

- ChromaDB : https://www.trychroma.com
- Qdrant : https://qdrant.tech
- FAISS : https://github.com/facebookresearch/faiss
- FastAPI : https://fastapi.tiangolo.com
- Pydantic : https://docs.pydantic.dev
- pyarabic : https://github.com/linuxscout/pyarabic

### Autres Ressources

- GTAF AI Stack (13M users) : https://gtaf.org/blog/f0-9f-9a-80-inside-gtafs-ai-stack-how-a-small-team-serves-13m-muslims-worldwide
- Daleel.app : https://daleel.app
- Tasleem AI : https://tasleem.ai
- Deen Buddy : https://deenbuddyapp.com
- Tarteel AI : https://tarteel.ai

---

*Document créé le 21 juin 2026. Toutes les décisions, raisons, et références capturées. Quiconque lit ce document peut comprendre le projet NUR dans son intégralité et le reconstruire from scratch.*

---

## 23. Points Non-Vérifiés / Risques

| # | Section | Problème | Action nécessaire |
|---|---------|----------|-------------------|
| 1 | Section 8 — Embeddings | ~~e5-large-instruct~~ remplacé par BGE-M3 | ✅ Résolu — BGE-M3 choisi et implémenté |
| 2 | Section 8 — Reranker | "Améliore de 40%" vient de blogs marketing | Mesurer le gain réel sur nos données islamiques |
| 3 | Section 6 — Hadith grades | Grades meeAtif non vérifiés académiquement | Croiser avec sunnah.com et vérifier cohérence |
| 4 | Section 12 — URLs | URLs sunnah.com/quran.com pas testées pour tous | Tester 50+ URLs |
| 5 | Section 17 — Hardware | Budget mémoire théorique | Mesurer RAM réelle sur M4 |
| 6 | Section 11 — Anti-Hallucination | Vérif caractère par caractère = concept | Implémenter avec Bi-level Matching + Levenshtein (recherche Gemini 2026) |
| 7 | Section 13 — IslamQA scraping | IslamQA peut bloquer | Vérifier API / utiliser mirrors |
| 8 | Section 5 — Cross-Refs | Pas de source existante | Utiliser Tafsir Ibn Kathir comme pont |
| 9 | Section 11 — NLI Verification | Module T4 nouveau — modèle NLI pas encore choisi | Tester DeBERTa-v3-large vs Ayn-NLI sur texte islamique arabe |
| 10 | Section 11 — Mawdu' Detection | Dataset ADAM-HA pas encore intégré | Télécharger ADAM-HA + indexer hadiths fabriqués connus |
| 11 | Section 7 — Groq Model | qwen3-32b deprecated juillet 2026 (alerte Gemini) | Planifier migration vers qwen3.6-27b |

---

## 24. Phases du Projet

### Phase 1 — Données + Embedding ✅ TERMINÉ
- Télécharger le Coran (AR + FR + EN, 6,236 ayahs) via **alquran.cloud API** (pas fawazahmed0 — moins fiable)
- Télécharger les hadiths (6 collections Kutub al-Sittah, 33,738 hadiths) via **meeAtif/hadith_datasets** sur HuggingFace (téléchargement direct URLs, pas le package `datasets`)
- Télécharger le Tafsir Ibn Kathir (AR + EN, 6,236 entrées chacun) via **spa5k CDN (jsDelivr)** — le HuggingFace parquet M-AI-C n'avait que 1,894/6,235 entrées EN
- Normaliser le texte arabe (supprimer diacritiques, normaliser أإآ→ا, ى→ي, ة→ه, supprimer tatweel)
- Chunking structure-aware : 1 chunk = 1 ayah (Coran) ou 1 hadith (Hadith) ou 1 section tafsir
- Embedding avec **BAAI/bge-m3 via FlagEmbedding** (dense + sparse + ColBERT en un seul `encode()`)
- Dense stocké dans ChromaDB : `quran_dense` (6,236), `hadith_dense` (33,738), `tafsir_ar_dense` (6,236), `tafsir_en_dense` (6,236) = **52,846 chunks total**
- Sparse stocké en JSON : `data/sparse/{source}_sparse.json` (ChromaDB open-source ne supporte PAS les sparse vectors)
- Hadith grades parsés : Sahihayn = Sahih par consensus, Mawdu/Munkar = `is_rejected=True`, `grade_weight=0.0` (gardé pour détection de faux hadiths)
- Embedding exécuté sur **Google Colab T4 GPU** (~40 min) au lieu de M4 MPS (~3h)
- **Résultat** : Base vectorielle hybride fonctionnelle, dense + sparse, testable avec des queries de similarité

### Phase 2 — Pipeline RAG + In-Context Isolation + Structured Output
- **Architecture code** :
  ```
  nur/src/
  ├── config.py          # pydantic-settings + .env
  ├── pipeline.py        # Classe NURPipeline (orchestrateur)
  ├── retriever/
  │   ├── dense.py       # ChromaDB queries (4 collections parallèles)
  │   ├── sparse.py      # Sparse JSON dot-product scoring
  │   └── fusion.py      # RRF fusion (k=25, α=0.4 dense / 0.6 sparse)
  ├── generator.py       # Groq API + Instructor + JSON Schema
  ├── sources.py         # Source ID Protocol + URLs clickables
  └── cli.py             # rich + prompt_toolkit
  ```
- Pipeline retrieval : query → BGE-M3 dense+sparse (SANS préfixe, ~50ms M4) → 4 queries ChromaDB parallèles + sparse scoring → RRF fusion → top-30
- **Connexion Groq API** : Qwen3-32B, temp=0.0, frequency_penalty=0.0, repetition_penalty=1.1
  - ⚠️ frequency_penalty=0.3 était TROP HAUT — peut corrompre les formules arabes répétitives (ﷺ, bismillah)
  - ⚠️ Planifier migration qwen3.6-27b (dépréciation qwen3-32b juillet 2026)
- **In-Context Isolation prompt** : interdiction FORMELLE d'utiliser les connaissances pré-entraînées
- **Format XML** pour les chunks : `<document id="SRC-QURAN-2-255"><source>...</source><content>...</content></document>` (-30% hallucinations vs texte brut)
- **Source IDs en majuscules** : `SRC-QURAN-2-255`, `SRC-HADITH-BUKHARI-1` (ancrage fort pour l'attention du LLM)
- **Chain-of-Thought Citation** via JSON Schema structuré :
  ```json
  {"thought_process": "...", "valid_source_ids": ["SRC-QURAN-2-255"], "synthesis": "..."}
  ```
- **JSON Schema** (PAS Tool Calling) via `response_format={"type": "json_schema", ...}` — constrained decoding natif sur LPUs Groq
- **Instructor** (compatible Groq via json_schema) pour validation + retry automatique
- **sunnah.com URLs** : stockées dans metadata Phase 1, NE PAS construire dynamiquement (numérotation décalée)
- **Détection langue** : `lingua-py` ou `fasttext` → force le LLM à répondre dans la langue de l'utilisateur
- **Retry** : `tenacity` exponential backoff + fallback Ollama qwen2.5:7b
- **Latence estimée** : ~1.2-1.8 sec sans reranker (50ms BGE-M3 + 40ms ChromaDB + 80ms sparse + 10ms RRF + ~1s Groq)
- **Résultat** : Chatbot fonctionnel en terminal, répond avec sources, zéro connaissances pré-entraînées

### Phase 3 — Reranking + Hybrid Search Fusion
- Implémenter Reciprocal Rank Fusion (RRF) pour combiner dense (ChromaDB) + sparse (JSON)
- Paramètres RRF : **k=25**, **α=0.4 dense / 0.6 sparse** (le sparse DOMINE pour texte islamique)
- Ajouter bge-reranker-v2-m3 comme cross-encoder reranker (`FlagReranker` ou `CrossEncoder`, ~1.2GB RAM, ~200ms/50 docs sur M4 MPS)
- **Top-30 retrieval → reranker → top-10** (5 était trop peu pour QA islamique multi-source)
- **⚠️ Pas de ColBERT** — redondant avec le reranker cross-encoder, gain <2%, coût disproportionné
- **Divergence Detection** : si reranker renvoie chunks opposés avec scores similaires → flag `Divergence=True`
- Comparer : retrieval seul vs hybrid vs hybrid+rerank sur un set de 20 questions test
- **Résultat** : Qualité de retrieval significativement améliorée + détection automatique d'ikhtilaf

### Phase 4 — Context-Enriched Chunks (Anthropic) — AVANCÉ depuis l'ancien Phase 6
- Pour chaque chunk, générer un préfixe contextuel via LLM (type du document, thème, cross-refs)
- Technique Anthropic : -67% d'échecs de retrieval
- Ré-embedder les chunks enrichis
- Comparer la qualité de retrieval avant/après sur le set de test
- **Résultat** : Meilleur matching sémantique grâce au contexte riche

### Phase 5 — Authenticity Weighting + Mawdu' Detection
- Implémenter le scoring pondéré par grade (Sahih +30%, Hasan +10%, Da'if -50%, Mawdu' weight=0.0)
- Mawdu/Munkar : gardés avec `is_rejected=True` — pour la DÉTECTION de faux hadiths
- **Base de données négative** : indexer les hadiths fabriqués connus (⚠️ ADAM-HA à vérifier — une source a confondu avec un dataset hate speech, pas hadith authentication. Chercher sur HuggingFace le dataset correct de hadith authentication arabe)
- **Stratégie Double-RAG** : base authentique + base fabriquée → avertissement automatique
- **Seuil de détection** : si cosine < 0.45 → "Ce texte ne figure pas dans les corpus authentiques"
- Ajouter l'affichage obligatoire du grade dans chaque citation hadith
- Ajouter un avertissement pour les hadiths hasan/da'if
- **Résultat** : Réponses éthiquement responsables, hadiths faibles signalés, faux hadiths détectés

### Phase 6 — Vérification Post-Génération (Anti-Hallucination Multi-Couches)
- **Module T4 — NLI Verification** : modèle DeBERTa-v3-large ou Ayn-NLI vérifie chaque phrase (entailment < 0.95 = rejet)
- **Module T5 — Bi-level Character Matching** pour l'arabe :
  - Niveau 1 : Smith-Waterman/Levenshtein après strip diacritiques + normalisation
  - Niveau 2 : comparaison stricte avec Coran de Médine/Uthmani — match 100% lettres de base
- **Module T6 — Decoupled Grounding** : `exact_citations` vs `synthesis` séparés, rejet au parser si citation pas mot-à-mot
- **Constrained Decoding** (Outlines/Instructor) : forcer le LLM à ne produire que des tokens valides pour les citations arabes
- Vérification des références hadith : collection + numéro existe-t-il ?
- Score de fidélité par affirmation
- **Résultat** : Zéro hallucination sur les versets coraniques, sources vérifiées par 3 couches indépendantes

### Phase 7 — Avis des Savants + Ikhtilaf
- Index des avis savants : collecter des fatwas d'IslamQA, Islamweb, Dar al-Ifta
- Structurer les avis avec : savant, madhhab, opinion, preuves, source livresque
- Détection d'ikhtilaf : quand plusieurs avis existent, les présenter avec leur niveau de consensus
- Implémenter les 5 niveaux : Ijma', majorité écrasante, désaccord significatif, avis isolé, aqeedah
- Principe de neutralité absolue : jamais encourager, jamais décourager
- **Résultat** : Réponses avec avis savants, ikhtilaf correctement présenté

### Phase 8 — Cross-References Coran↔Hadith
- Utiliser le Tafsir Ibn Kathir comme pont : pour chaque ayah, extraire les hadiths référencés
- Construire la Cross-Reference DB : ayah → hadiths, hadith → ayahs
- Dans les réponses, quand on cite un verset → proposer les hadiths qui l'expliquent, et inversement
- **Résultat** : Réponses riches avec connexions Coran↔Hadith↔Tafsir

### Phase 9 — Interface Web (Next.js PWA)
- Frontend **Next.js PWA** — pas React Native (pas de $100 dev account, testable sur PC+mobile, déploiement gratuit Vercel)
- Affichage riche : texte arabe (grand, droite→gauche) + traduction FR + grade + lien clickable
- Toggle de langue (AR/FR/EN)
- Affichage des sources en accordion
- Mode clair/sombre
- PWA = installable sur mobile comme une app native, sans App Store
- **Résultat** : Application web utilisable dans le navigateur + installable mobile

### Phase 10 — Advanced RAG + CRAG
- Corrective RAG : évaluateur de retrieval qui grade les documents avant génération
- Query decomposition : questions complexes → sous-questions
- Adaptive routing : classifier la question (factuelle, fiqh, aqeedah) et adapter le pipeline
- Fallback Groq → Ollama local si APIs down
- **Résultat** : Pipeline RAG robuste et intelligent

### Phase 11 — Évaluation + Benchmarking
- Créer un dataset d'évaluation : 50-100 questions avec réponses attendues et sources
- Implémenter RAGAS (faithfulness, answer relevancy, context precision/recall)
- Métriques islamiques : source attribution accuracy, hadith grading accuracy, ikhtilaf awareness, abstention rate
- Comparer les phases : baseline (Phase 2) vs chaque amélioration
- **Résultat** : Mesures objectives de qualité, identification des points faibles

### Phase 12 — Features Avancées
- Input vocal via Groq Whisper Large v3
- TTS pour les résumés/explications (Orpheus Arabic Saudi pour l'interface, PAS pour le Coran)
- Prompt Guard contre les injections
- Quran ontology / Graph RAG (si la phase 8 justifie le besoin)
- Mode offline complet via Ollama
- **Résultat** : Application complète et robuste
