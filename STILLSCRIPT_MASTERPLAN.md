# StillScript Confidential Transcripts — Meester Ontwikkelingsplan

**Plan-weergawe:** 1.9
**Sagteware-teikenweergawe:** v4.0.0 (eerste publieke vrystelling)
**Huidige weergawe:** v3.1.0 (Fase 2 beta — intern / vriende & familie)
**Laas opgedateer:** 2026-07-28
**Eienaar:** Danie (Acutus Consulting)

---

## Doel

Van v3.1.0 (werkende twee-modus beta, DanScribe-naam) na v4.0.0 — die eerste
publieke vrystelling onder die naam **StillScript Confidential Transcripts**,
met 'n werkende Akkuraat-modus, volledige dokumentasie, en 'n
vertroulikheidsposisionering gemik op prokureurs, sielkundiges, beraders en
joernaliste.

**Leidende beginsel:** geen dringendheid, 'n ordentlike ding. Onafhanklike take
met eksterne wagtyd word EERSTE afgehandel sodat niks die vrystelling aan die
einde bottelnek nie.

---

## Weergawebeheer-skema (semver)

| Weergawe | Betekenis |
|---|---|
| `v3.1.0` | Huidige beta (Fase 2). Reeds gebou, intern getoets. |
| `v4.0.0-beta.N` | Ontwikkelingsbou tydens Golf 2–4 (voorvrystelling-etikette). |
| `v4.0.0` | Publieke lansering. Naam + Akkuraat-modus + dokumentasie volledig. |
| `v4.0.x` | Regstellings ná lansering (bugfixes). |
| `v4.1.0` | Klein kenmerke ná lansering (geen breukverandering). |
| `v5.0.0` | Breukverandering (indien ooit). |

Elke etiket sneller die GitHub Actions Windows-bou. Konsep-Release totdat
publiek gereed.

---

## Status-legende

⬜ Nie begin nie   🟡 Aan die gang   🔴 Geblokkeer (wag op afhanklikheid)   ✅ Klaar

---

## GOLF 0 — Reeds afgehandel (verwysing)

| # | Item | Status | Notas |
|---|---|---|---|
| 0.1 | Fase 1: transkripsie-naat + lengte-gedrewe diarisering-hibried | ✅ | `f9fe204`, `4522455` |
| 0.2 | Fase 2: twee-modus-UI, herkoms-metadata, krediete-oppervlak | ✅ | v3.1.0 |
| 0.3 | Windows CI-bou (ffmpeg gebundel, CPU-torch, PyInstaller-versameling) | ✅ | groen bou, konsep-Release |
| 0.4 | `torch==2.13.0+cpu` in CI gesluit | ✅ | bevestig uit eerste groen bou |
| 0.5 | Akkuraat-modus dekodering-konfig geverifieer | ✅ | large-v3, direkte `generate()`, `condition_on_prev_tokens=True` |
| 0.6 | Branding bevestig: geen CIPC/handelsmerk-konflik, domeine beskikbaar | ✅ | "StillScript Confidential Transcripts" |

---

## GOLF 1 — Onafhanklike / wagtyd / besluit-items (DOEN EERSTE — kan parallel loop)

Hierdie items hang NIE van die tegniese bou af nie. Front-load hulle sodat
eksterne wagtyd en ontwerpwerk nie later die vrystelling blokkeer nie.

| # | Item | Status | Afhanklikheid | Notas |
|---|---|---|---|---|
| 1.1 | **André-bevestiging: watter HF-repo is kanoniek** (hoof `whisper-large-v3-afrikaans` vs `lora-v1`) | 🔴 wag op André | Ekstern | Reeds gevra op HF-draad. Blokkeer VERSKEEP van akkuraat-enjin, nie bou/toets nie. Monitor. |
| 1.2 | **Vertroulikheidsverklaring opstel** (teks) | ✅ | Geen | Opgestel: `STILLSCRIPT_CONFIDENTIALITY.md`. Transkripsie 100% plaaslik; geen tracking; opsionele Claude-opsomming die enigste uitsondering (opt-in, gebruiker se eie API-sleutel, net teks). *Plasing* in installer = Golf 4.3. Formele nakoming-hersiening (POPIA) bly aparte stap deur gekwalifiseerde persoon. |
| 1.3 | **Nuwe logo** | ✅ | Geen | Produksie-lêers ontvang van Claude Design: ware vektor SVG's (primêre horisontale logo, primêre simbool, mikro-simbool vir 16-32px, monochrome swart/wit/navy-variante), egte multi-resolusie `.ico` (16/32/48/256px), volledige PNG-stel (16/24/32/48/64/128/256/512px). Klein-groottes self bevestig op werklike skerm (100%-zoem). Handelsmerk-riglyn-dokument (`STILLSCRIPT_CONFIDENTIALITY.md`-styl brand guide, 5 bladsye) ook ontvang as formele riglyn-verwysing. Konsep: "The Settling Line". Palet: Ink Navy #0D1B2A / Slate #44566B / Sage #6B6F58 / Terracotta #8A5647. Tipografie: Cormorant Garamond (wordmark) + Source Sans 3 (liggaam). Gereed vir Golf 4.2 (inbou in app/installer). |
| 1.4 | **Reproduseerbaarheid-besluit** | ✅ besluit | Geen | **BESLUIT: laat die gebruiker kies, in Instellings, met gevolg-taal (nie "temperature" nie).** Twee opsies: "Consistent" (temperature=0 — selfde oudio gee selfde transkripsie; beste vir ouditeerbare rekords) en "Best effort" (verstek-terugval — herprobeer moeilike segmente vir maksimum akkuraatheid; herhaalde transkripsies kan effens verskil). **Verstek = "Consistent"** (pas by oudit/herkoms-belofte aan regsmark). Implementering → Golf 2. **MOET beklemtoon word in User Manual (Golf 5.2).** |

---

## GOLF 2 — Tegniese kern: Akkuraat-modus bou

| # | Item | Status | Afhanklikheid | Model (CC) | Notas |
|---|---|---|---|---|---|
| 2.1 | Large-v3 Afrikaanse enjin agter Akkuraat-branch inprop (direkte `transformers.generate()`, `condition_on_prev_tokens=True`) | ✅ | Golf 0.5 ✅; 1.4 besluit | Opus 5 | **Klaar.** `accurate_engine.py` geskep, `transcribe_audio_accurate()` as parallelle inskrywingspunt in `DanScribe_v2.py` (+27 reëls, suiwer additief). 18/18 toetse geslaag; `condition_on_prev_tokens=True` bevestig deur die werklike `generate()`-oproep te bespied. Fast mode bewys onaangeraak. Modelpad: `~/whisper_afrikaans_spike/merged_afrikaans_fp32/` (oorskryfbaar via `STILLSCRIPT_ACCURATE_MODEL_DIR`). Temperatuur staan op die Golf 0.5-ladder (nie `0.0` nie) — albei konstantes (`TEMPERATURE_CONSISTENT`/`TEMPERATURE_BEST_EFFORT`) reeds bygevoeg vir 'n eenreël-verandering in 2.9. `language="af"` hardgekodeer (geen outo-detect). |
| 2.1a | Verpakkingstrategie — ondersoek + besluit | ✅ besluit | 2.1 ✅ | Opus 5 (ondersoek) | **BESLUIT: opsie (b) — aflaai-op-aanvraag, gehuisves op Hugging Face Hub** (kan later na Cloudflare R2 skuif indien volume/koste dit regverdig). Ondersoek het getoon: torch loop reeds in die 317 MiB-exe (via `openai-whisper`), dus die ware afhanklikheid-delta is net ~36–40 MiB gekomprimeer (transformers/tokenizers/huggingface_hub/safetensors). Model = 5.75 GiB rou / ~3.69 GiB gekomprimeer (fp32-gewigte grootliks fp16-waardes verbreed). GitHub Releases se 2 GiB-per-lêer-limiet bevestig — blokkeer bundel-opsies (a)/(c) permanent. |
| 2.1a.1 | Empiriese CI-validasie van installer-grootte | ⬜ | 2.1a ✅ | Sonnet 5 | Bevestig die ~355 MiB-skatting met 'n regte bou (transformers + safetensors bygevoeg, torch NIE — reeds daar), nie net rekenkunde nie. Goedkoop, lae risiko. **Let wel (uit 2.1a.2):** `requirements.txt` het nog nie `transformers`, `safetensors` óf `huggingface_hub` nie — al drie word nou by loopyd benodig (enjin, guard, aflaaier). Pin hulle as deel van hierdie item; `huggingface_hub` moet `>=1.5.0,<2.0` wees (transformers 5.14.1 se eis) en die weergawe waarteen 2.1a.2 getoets is, is 1.24.0.<br><br>**BEWAAR UIT DIE WEGGOOI-TAK (nou uitgevee — sien changelog 1.9).** Die weggooi-tak `wave2.1a1/size-investigation` (commit `02ce32a`) het die werklike gemete pins gehou. Hulle word hier bewaar sodat die tak veilig verwyder kon word — hierdie is die vertrekpunt vir hierdie item, nie afleidings nie, maar 'n werklike `pip install`:<br>`transformers==5.14.1`, `safetensors==0.8.0`, `tokenizers==0.22.2`, `huggingface_hub==1.25.1`, `hf-xet==1.5.2`, `PyYAML==6.0.3`, `click==8.4.2`, `typer==0.27.0`, `rich==15.0.0`, `pygments==2.20.0`, `markdown-it-py==4.2.0`.<br>**Belangrike bevinding uit daardie meting:** `huggingface_hub` sleep sy eie CLI-stapel saam (typer/rich/click/pygments/…) al gebruik die app dit nooit — ~9 MiB gekomprimeer wat die oorspronklike 2.1a-rekenkunde glad nie in ag geneem het nie. Oorweeg om dit uit te sluit in die PyInstaller-stap.<br>**Let op die botsing:** die tak pin `huggingface_hub==1.25.1`, maar 2.1a.2/2.1a.3 is teen **1.24.0** gebou en getoets. Kies bewustelik een en toets weer — moenie die verskil stilweg laat inglip nie.<br>Die tak het ook `--collect-all transformers` + `--collect-all safetensors` by die CI-bou gevoeg; dit was net nodig omdat `DanScribe_v2.py` destyds nog nie `accurate_engine` ingevoer het nie. Sodra 2.3 die enjin werklik inprop, maak die invoer self daardie twee reëls oorbodig.<br><br>**Opruiming klaar (2026-07-28):** die weggooi-tak `wave2.1a1/size-investigation` is van origin én plaaslik verwyder, en die konsep-vrystelling `v0.0.0-wave2.1a1-sizetest` is uitgevee. Daar was **geen git-tag** om te verwyder nie — die konsep-vrystelling was "untagged" (GitHub-plekhouer `untagged-3885b548…`), dus is `v0.0.0-wave2.1a1-sizetest` nooit as 'n regte tag gepubliseer nie. Geen werkvloei-lêer het daarna verwys nie (`build-windows-release.yml` sneller op die generiese `v*.*.*`). `main` is heeltemal onaangeraak (`62b0b4e`, plaaslik = origin), en die oorblywende tags is net `v3.0.0`, `v3.0.1`, `v3.1.0`. |
| 2.1a.2 | Bou aflaai-op-aanvraag-meganisme (HF Hub, verifieer via `full_tensor_digest`) | ✅ | 2.1a.1 | Opus 5 | **Klaar.** `accurate_model_download.py` geskep — een inskrywingspunt vir 2.3: `ensure_accurate_model(model_dir=None, progress_callback=None, force=False) -> str`, plus `describe_download()` vir die toestemming-dialoog (grootte/lêertal vooraf). Geen UI; niks aan die Akkuraat-knoppie geraak nie. **Pad:** `~/.danscribe_models/accurate-af-large-v3/` — volg die app se bestaande `~/.danscribe*`-konvensie (`~/.danscribe.log`, `~/.danscribe_config.json`), sodat 4.1 se hernoem-sweep dit outomaties vang. **Revisie gepin** op die volle 40-karakter sha `fedc2529…` (nie `main` nie), dieselfde redenasie as die guard se gepinde vingerafdruk. **Volledige verifikasie onvoorwaardelik** ná elke vars aflaai (`verify_merged_model(..., full=True)` — die eksplisiete argument klop die omgewingsveranderlike), want truncation/korrupsie is presies wat die 9-monster-verstek mis. Guard-mislukking → gids uitgevee + "probeer weer"-boodskap; nooit 'n gids wat die gebruiker self aangewys het nie. Voltooiing-stempel `.stillscript_download.json` verhoed dat 'n halwe aflaai later stilweg aanvaar word. 93/93 toetse geslaag (`test_accurate_model_download.py`); 2.2 se guard-toetse steeds groen. **Werklike grootte is 5.76 GiB, nie 3.69 GiB nie** — sien waglys. |
| 2.1a.2a | Xet-terugval (outomaties, geen gebruikersaksie) | ✅ | 2.1a.2 | Opus 5 | Ingebou in 2.1a.2 en werklik getoets (nie net verdedigend geskryf nie). **Belangrike tegniese bevinding:** `HF_HUB_DISABLE_XET=1` in `os.environ` sit is te laat — `huggingface_hub.constants` lees dit *een keer, by invoer*. Die module stel dus die **module-attribuut** `constants.HF_HUB_DISABLE_XET` (dít is wat werk) én die omgewingsveranderlike. Bevestig met `is_xet_available()` binne/buite die blok. Klassifikasie is doelbewus nou: skyf vol (ENOSPC), toestemming, en "repo/revisie bestaan nie" gaan reguit deur na die gebruiker; net vervoerfoute word herprobeer, en dan word eers getoets of huggingface.co self bereikbaar is sodat "geen internet" nie as 'n Xet-probleem aangebied word nie. |
| 2.1a.3 | **Hervatbare aflaai — model in ~200 MiB stukke opgedeel** | ✅ | 2.1a.2 ✅ | Opus 5 | **Klaar.** **BESLUIT was: opdeel, nie 'n eie ranged-HTTP-lus nie** — hergebruik lêer-vlak-hervatting (wat reeds bewese werk) eerder as om self 'n vervoerpad te skryf vir 'n model waarvan die waarde op verifieerbaarheid rus. `shard_accurate_model.py` geskep: greep-vlak (NIE tensor-bewus nie — herserialisering sou die grepe verander en die gepinde vingerafdruk ongeldig maak) opdeling in 30 stukke + `manifest.json` (volgorde, grootte, sha256 per stuk, plus sha256 van die hele hersaamgestelde lêer). Plaaslike heen-en-weer-toets bevestig: hersamestelling gee `33bdc94e…`, presies die gepinde model. `accurate_model_download.py` het nou `layout="chunked"` (verstek) langs `layout="single"` (2.1a.2 se pad, behou as terugvalopsie). Elke stuk word by aankoms teen die manifes geverifieer; die hersaamgestelde lêer teen die manifes se heel-lêer-sha256; dán die guard in volledige modus — drie onafhanklike kontroles. **Stukke word NIE uitgevee as die guard misluk nie** (net die slegte hersaamgestelde lêer), sodat 'n herprobeer minute vat i.p.v. ure. Skyfbehoefte tydens installasie styg na ~11.5 GiB (stukke + hersaamgestelde lêer gelyktydig), settle op 5.75 GiB — `_check_disk_space()` weier vooraf i.p.v. die skyf half vol te maak. **Terugval-item:** verwyder die ou monolitiese `model.safetensors` uit die HF-repo eers nadat die stuk-pad in produksie bevestig is — sien waglys.<br><br>**Gepubliseer:** commit `6baf2473d04da504f039ade149512d891e4a7ca5` (44 lêers = 13 oorspronklik + 30 stukke + manifes; die monoliet is ONAANGERAAK). `CHUNKED_REVISION` in `accurate_model_download.py` wys hierheen; `REVISION` (`fedc252…`) bly die enkellêer-pin. `EQUIVALENT_REVISIONS` keer dat 'n bestaande 2.1a.2-installasie 5.75 GiB oor niks herlaai nie — albei commits dra dieselfde grepe.<br>**Werklik getoets teen die regte repo (nie gesimuleer nie):** (3a) volle stuk-aflaai voltooi; hersamestelling in 41.5s; sha256 `33bdc94e…` = die gepinde model; guard FULL geslaag (9 probes + al 1259 tensors); stukke daarna outomaties verwyder (5.75 GiB teruggewin). (3b) SIGKILL middel-stuk: 630 MiB gevorder, 600 MiB behou, **net 30 MiB heroordra teenoor 630 MiB vir 'n volle herbegin — 95.2% behou**. (3c) Xet-terugval per stuk: werklike 200 MiB-stuk oor die HTTPS-terugval gehaal ná 'n ingespuite Xet-fout, sha256 korrek. (3d) bederfde stuk word bespeur, uitgevee en heraflaai (200 MiB i.p.v. 5888 MiB); by 'n guard-mislukking word slegs die hersaamgestelde lêer uitgevee en **al 30 stukke behou**. 135/135 toetse; guard- en enjin-toetse steeds groen.<br>**Bug wat net die werklike lopie gevang het:** die stuk-pad het die *lêerpad* i.p.v. die *gids* teruggegee, wat 2.3 se eerste oproep sou breek. Reggestel + regressietoets vir albei uitlegte bygevoeg. |
| ~~2.1a.3 (oorspronklike voorstel)~~ | ~~eie ranged-HTTP lus~~ | ❌ verwerp | — | — | **Nuut ontdek in 2.1a.2, deur werklike meting.** `huggingface_hub` 1.24.0 hervat **per lêer, nie per greep nie**. 'n Netwerk-hik *binne* een lopie is veilig (`http_get` herprobeer met `Range`), maar as die proses doodgaan (toemaak, crash, kragonderbreking) begin daardie lêer weer op nul: hf_hub skryf na 'n proses-unieke `<etag>.<uuid>.incomplete` en vee dit in 'n `finally` uit (PR #4228). Getoets: 56 MiB oorgedra, SIGKILL, herbegin → weer van nul af. Vir hierdie repo is dit erg, want `model.safetensors` (5.75 GiB) is 99.9% van die vrag — die 12 klein lêers hervat, die een wat saak maak nie. Teen die gemete ~0.6 MiB/s is dit ~2.7 uur wat verlore gaan as die gebruiker die app toemaak. **'n Ouer huggingface_hub is nie 'n opsie nie:** transformers 5.14.1 vereis `huggingface-hub>=1.5.0,<2.0`, en die per-greep-hervatting is voor 1.5 verwyder. Oorblywende opsie: haal die een groot lêer met ons eie `Range`-lus. Dit beteken egter dat ons die vervoerpad self skryf vir 'n model waarvan die hele waarde op verifieerbaarheid rus — dus 'n besluit vir Danie, nie 'n stille implementeringskeuse nie. Die guard se volledige verifikasie bly die vangnet ongeag watter pad gekies word. |
| 2.1a.4 | **Modelkaart (README) op die publieke HF-repo** | ✅ | 2.1a.3 ✅ | Opus 5 | **Klaar (2026-07-28), commit `ac426e10`.** Die repo was publiek sonder enige modelkaart — 'n werklike erkenningsgaping vir 'n afgeleide van André Oosthuizen se werk. **Lisensie is by die bron geverifieer, nie uit ons eie notas aanvaar nie:** `andreoosthuizen/whisper-large-v3-afrikaans` is **CC-BY-4.0** (bevestig op drie plekke — HF-etikette, die YAML-frontmatter, en die kaart se liggaam: *"Creative Commons Attribution 4.0 (Commercial use allowed)"*). Daar is géén aparte `LICENSE`-lêer nie; die modelkaart self is die lisensieverklaring. Die datastel `andreoosthuizen/afrikaans-30s` is ook CC-BY-4.0, en die basismodel `openai/whisper-large-v3` is **Apache-2.0**. Die saamgesmelte lêer dra albei bydraes onskeibaar, dus is ons kaart onder **CC-BY-4.0** gepubliseer (die strenger een vir 'n herverspreider) met Apache-2.0 uitdruklik erken vir die basisgewigte. **Let op:** CC-BY-4.0 het géén ShareAlike-klousule nie, dus *dwing* dit ons nie om dieselfde lisensie te gebruik nie — ons pas dit aan omdat dit 'n herverpakking is, nie 'n nuwe werk met eie terme nie. André se BibTeX-aanhaling word woordeliks weergegee. Die kaart dokumenteer ook die dekodeerder-alleen-bevinding uit 2.2 (64 enkodeerder-`lora_B`-tensors presies nul) as bruikbare deursigtigheid vir ander, en verduidelik die twee uitlegte. **Die gepinde revisies `fedc252…` en `6baf2473…` los steeds op (HTTP 200)** — die kaart is 'n nuwe commit bo-op hulle en raak nie die app se aflaaipad nie. Dra by tot 2.7 se CC-BY-verpligting, maar vervang dit nie: die in-app krediete-inskrywing bly steeds nodig. |
| 2.2 | Adapter-guard by opstart (waarde-gebaseerde toets, nie net nie-nul tensor-telling nie) | ✅ | 2.1 ✅ | Opus 5 | **Klaar.** Meesterplan se oorspronklike "nie-nul tensor-telling"-voorstel sou op 'n stock large-v3-model óók geslaag het — Opus 5 het 'n vierlaag-wag gebou i.p.v.: struktuur, fyn-afregting-bewys (6 dekodeerder-tensors, ≥1e-3 drempel), afkoms (3 tensors wat 'n q/v-LoRA nooit kan raak nie), identiteit (sha256 per monster). **Belangrike bevinding:** André se adapter se enkodeerder-`lora_B`-tensors (64 stuks) is almal presies nul — nooit afgerig nie; slegs die dekodeerder se q/v_proj (128 tensors) is werklik Afrikaans-aangepas. Getoets teen die werklike basismodel op skyf. Agterdeur (`STILLSCRIPT_ACCURATE_GUARD=evidence`) is verwyder — geen manier om die wag by loopyd te verswak nie. Opt-in volledige verifikasie (`full_tensor_digest`, hash al 1259 tensors) bygevoeg: 5.25s koud / 3.70s warm, teenoor 1.15s/0.05s vir die 9-monster-verstek. 56/56 toetse geslaag; Golf 2.1 se 18/18 steeds groen. |
| 2.2a | Verstel volledige verifikasie na verstek (nie opt-in nie) | ⬜ | 2.2 ✅ | Sonnet 5 | Danie het ingestem: ~4s koue-koste teen 'n operasie wat ~3× regtyd loop is verwaarloosbaar, en "ons verifieer die hele modellêer" is 'n regte verkoopspunt vir die regsmark. Tans nog agter `STILLSCRIPT_ACCURATE_FULL_VERIFY`. Eenreël-verandering — moenie laat glip nie. |
| 2.3 | Akkuraat-knoppie aktiveer (verwyder "Coming soon") | ⬜ | 2.1 ✅; 2.1a.2 ✅ | Sonnet | UI reeds gebou in Fase 2; net aktiveer. Blokkasie opgehef — 2.1a.2 is klaar. **Hoe om dit te roep:** `path = ensure_accurate_model(progress_callback=hook)` en gee dan `accurate_engine.transcribe(audio, model_dir=path)`. Roep dit elke keer wat Akkuraat gekies word — as die model reeds afgelaai en geverifieer is, keer dit onmiddellik terug sonder om die netwerk te raak. Wys eers `describe_download()` se grootte/tyd voordat enigiets begin (sien waglys: moenie stilweg 5.76 GiB begin trek nie). `progress_callback` word van **een agtergronddraad** af geroep — marshal self na Tk se hoofdraad. Vang `AccurateModelDownloadError`: die `str(e)` is reeds in gebruikerstaal geskryf en kan direk in 'n `messagebox` gaan; `e.can_retry` sê of 'n "Probeer weer"-knoppie sin maak. |
| 2.4 | Tydswaarskuwing + vorderingsaanduiding (~3× regtyd, bondel-modus) | ⬜ | 2.1 ✅ | Opus 5 | Gebruiker moet vooraf weet dit is 'n oornag-taak. Steady-state ~3× (Toets3-langvorm-maatstaf) — die 2.1-toets se 4.25× sluit koue modellaai in, nie die regte getal om aan te haal nie. |
| 2.5 | Geheue-bestuur: Medium uit geheue laat val voor large-v3 laai | ⬜ | 2.1 ✅ | Opus 5 | ~8.7 GB piek; interakteer met diarisering-hibried. |
| 2.6 | Herkoms-blok uitbrei (model-ID + adapter-revisie-SHA vir Akkuraat) | ⬜ | 2.1 ✅ | Sonnet | Naat reeds ontwerp om dit by te voeg sonder herstruktuur. |
| 2.7 | Krediete: André se model + datastel byvoeg (CC-BY-4.0 erkenning) | ⬜ | 2.1 ✅ | Sonnet | Verplig onder lisensie. Krediete-oppervlak reeds gebou; net inskrywing byvoeg. **Lisensie bevestig in 2.1a.4:** model én datastel is CC-BY-4.0 (basismodel Apache-2.0). Die HF-modelkaart dek nou die *herverspreiding*-kant; hierdie item bly oop vir die *in-app* erkenning, wat 'n aparte verpligting is. Gebruik dieselfde bewoording en André se BibTeX uit die modelkaart sodat die twee nie uitmekaar dryf nie. |
| 2.8 | Akkuraat-modus toets op moeilike + maklike Afrikaanse oudio | ⬜ | 2.1–2.7 | — (jy toets) | 2.1 se rooktoets (30s, een spreker) was bemoedigend — koherente, korrek-gespelde Afrikaans, geen Nederlandse-drif — maar bewys nog niks op moeilike/multi-spreker-oudio nie. Let op die twee condition-stotters. **Belangrik vir die Fast-Mode-scope-besluit (2.11).** |
| 2.9 | Reproduseerbaarheid-keuse bou (Instellings: "Consistent"/"Best effort", verstek Consistent) | ⬜ | 1.4 ✅; 2.1 ✅ | Opus | Engine reeds voorbereid met albei konstantes (`TEMPERATURE_CONSISTENT`/`TEMPERATURE_BEST_EFFORT`) — eenreël-verandering. |
| 2.10 | Meet werklike akkuraatheidsverskil temperature=0 vs. terugval op moeilike oudio | ⬜ | 2.1 ✅ | Sonnet | Bevestig hoeveel swakker "Consistent" werklik is, sodat die verstek 'n ingeligte keuse is. |
| 2.11 | Besluit Fast Mode se toekomstige scope (Engels-verstek? waarskuwing vir nie-Engelse tale?) | 🔴 wag op 2.8 | 2.8; 2.1a ✅ | — (besluit, nie CC nie) | Fast Mode (Medium) se Nederlandse-drif-probleem lyk Afrikaans-spesifiek, nie universeel nie. Wag op 2.8 se volle validasie voor finale besluit. Waarskynlike uitkoms: Fast Mode bly verstek vir Engels/vinnige konsepte; Akkuraat word verstek/verpligtend vir Afrikaans. |

---

## GOLF 3 — Installer-opruiming + reproduseerbaarheid-implementering

| # | Item | Status | Afhanklikheid | Model (CC) | Notas |
|---|---|---|---|---|---|
| 3.1 | Fix `DanScribe_v2.iss` hardgekodeerde `C:\Users\danie\` paaie | ⬜ | Geen | Sonnet | Nie deur CI geloop nie; aparte taak. Maak relatief. |
| 3.2 | Reproduseerbaarheid implementeer | ⬜ | 1.4 besluit; 2.9 | Opus | Raak beide modusse se transkripsie-konfig. |

---

## GOLF 4 — Naam-oorgang (gekoördineer — raak baie oppervlakke)

| # | Item | Status | Afhanklikheid | Model (CC) | Notas |
|---|---|---|---|---|---|
| 4.1 | Hernoem produk na "StillScript Confidential Transcripts" oor alle oppervlakke (UI-titel, exe-naam, installer, konfig-paaie) | ⬜ | Golf 2 klaar | Opus | Een gekoördineerde stap. Let op konfig/log-paaie (`~/.danscribe*`). |
| 4.2 | Nuwe logo inbou | ⬜ | 1.3 ✅ | Sonnet | Vervang `logo.jpg`, `danscribe.ico`. |
| 4.3 | Vertroulikheidsverklaring in installer plaas | ⬜ | 1.2 ✅ | Sonnet | Uit Golf 1.2 se teks. Voeg eksplisiete verduideliking by dat die Akkuraat-modelaflaai (2.1a.2) net gewigte haal, geen gebruikersdata stuur nie. |

---

## GOLF 5 — Dokumentasie (LAASTE — beskryf finale produk)

| # | Item | Status | Afhanklikheid | Model (CC) | Notas |
|---|---|---|---|---|---|
| 5.1 | README volledig opdateer (Fast/Accurate-modus, nie ou Base/Small/Medium nie) | ⬜ | Golf 2 + 4 | Sonnet | Beskryf finale produk, dus laaste. Moet die HF Hub-aflaai vir Akkuraat-modus noem. |
| 5.2 | User Manual volledig opdateer | ⬜ | Golf 2 + 4 | Opus | Beskryf twee modusse, vertroulikheid, verwagte hersiening vir name. **MOET die reproduseerbaarheid-keuse (1.4/2.9) prominent verduidelik**, sowel as die Akkuraat-modus-aflaai (grootte, tyd, dat dit net gewigte is — geen oudiodata nie). |

---

## GOLF 6 — Vrystelling

| # | Item | Status | Afhanklikheid | Notas |
|---|---|---|---|---|
| 6.1 | Bou v4.0.0, toets op Windows-masjien | ⬜ | Alle bogenoemde | Vriend se masjien / VM voor publiek. |
| 6.2 | Landingsblad + verspreiding-vloei (Gumroad/Paddle) | ⬜ | 6.1 | Prys-vlakke: solo-praktisyn vs. praktyk. Vertroulikheidsverklaring op blad. |
| 6.3 | Publieke lansering | ⬜ | 6.1, 6.2 | Eerste betalende gebruikers. |

---

## Waglys / Bekende risiko's (monitor, nie take nie)

| Risiko | Status | Aksie |
|---|---|---|
| Diarisering etiket-verskuiwing (~0.7%) op lang lêers via temp-WAV-pad | Hanteer deur lengte-gedrewe hibried (drumpel 20 min) | Heroorweeg slegs indien 'n spesifieke kliënt presiese spreker-akkuraatheid op lang lêers vereis. |
| Twee condition-stotters in Akkuraat-modus ("jy het jy het", "gevra het gevra het") | Klein, nie-spiralerend | Hou dop op ander oudio tydens 2.8-toetsing. Nie 'n blokkasie nie. |
| Akkuraat-modus latensie ~3× regtyd | Verwag, bondel/oornag-modus | Posisioneer eerlik as "stel op, kom later terug". |
| Whisper nie-determinisme (temperatuur-terugval, steekproefneming) | Opgelos deur besluit 1.4 | Gebruiker kies "Consistent" (reproduseerbaar) of "Best effort". Verstek Consistent. Bou = 2.9; verduidelik = 5.2. |
| André antwoord dalk nie / gee onduidelike antwoord | Wag | Terugval: hoof-repo se `last-checkpoint`-gewigte is geverifieer as die 12.85%-model. Kan daarop verskeep indien nodig, met erkenning. |
| Akkuraat-modelaflaai vereis internetkonneksie by eerste gebruik | Aanvaar as gevolg van 2.1a se besluit | Moet duidelik gekommunikeer word in UI (2.3) en User Manual (5.2) voor die aflaai begin — nie stilweg nie. `describe_download()` gee die syfers. |
| **Aflaai-grootte is 5.76 GiB, nie die geskatte 3.69 GiB nie** | Bevestig in 2.1a.2 teen die werklike HF-repo | Die "~3.69 GiB gekomprimeer"-syfer uit 2.1a was 'n kompressie-skatting; die werklike oordrag is nie so gekomprimeer nie. **Gebruik 5.76 GiB in alle gebruikerstaal** (UI 2.3, handleiding 5.2, installer 4.3). Xet-deduplisering kan die drade-getal effens verlaag, maar moenie daarop staatmaak nie. |
| ~~'n Onderbreekte aflaai begin die groot lêer van vooraf~~ | ✅ **opgelos deur 2.1a.3** | Model is nou in 30 × ~200 MiB stukke. 'n Onderbreking kos hoogstens een stuk (~6 min teen 0.6 MiB/s) i.p.v. ~2.7 uur. 2.3 se UI hoef nie meer te smeek dat die app oopgelos word nie. |
| **Verwyder die ou monolitiese `model.safetensors` uit die HF-repo** | ⬜ **opvolg-item uit 2.1a.3** | Die stuk-uitleg is BYGEVOEG langs die oorspronklike lêer; niks is uitgevee nie, sodat 2.1a.2 se pad as terugval bly werk. Verwyder die monoliet eers nadat die stuk-pad op 'n regte installasie (Golf 2.3/6.1) bevestig is. Wanneer dit gebeur: verwyder terselfdertyd `layout="single"`, die `REVISION`-konstante en `EQUIVALENT_REVISIONS` uit `accurate_model_download.py` — hulle hoort saam in een commit. Let op: die repo dra tans albei uitlegte, dus ~11.5 GiB gehuisves; die monoliet verwyder halveer dit. |
| **Oplaai na HF is baie stadiger as aflaai: ~0.26 MiB/s gemeet (≈7 uur vir 5.75 GiB)** | Gemeet 2026-07-28 by die netwerkkoppelvlak, oor wifi | Asimmetriese lyn. Raak net Danie se eenmalige publiseerstappe, nie gebruikers nie. As 'n hersharding ooit weer nodig is: gebruik 'n bekabelde verbinding, en verwag ure. **Xet help NIE hier nie** — sien volgende ry. |
| **Xet se deduplisering werk nie vir die stukke nie** | Getoets 2026-07-28: 29 van 37 `query_dedup`-oproepe was "cache miss" | Die verwagting was dat greep-snitte van 'n lêer wat reeds in die Xet-CAS is, byna gratis sou oplaai. Dit gebeur nie. Boonop het die Xet-oplaaipad presies die bekende `cas::upload_xorb`-verbindingsfoute getref (4 keer in een lopie). Gebruik `--no-xet` vir enige toekomstige publisering. |
| **Skyfbehoefte tydens installasie is ~11.5 GiB, nie 5.75 GiB nie** | Gevolg van 2.1a.3 se ontwerp | Die stukke en die hersaamgestelde lêer bestaan gelyktydig, want die stukke word doelbewus behou tot ná die guard geslaag het (sodat 'n mislukking minute kos, nie ure nie). Settle op 5.75 GiB. `_check_disk_space()` weier vooraf met 'n duidelike boodskap. **Moet in die handleiding (5.2) en die installer (4.3) genoem word** — 'n ou skootrekenaar met 15 GiB vry is 'n werklike geval. |
| Xet se vordering beweeg in spronge — die persentasie kan **minute lank** stilstaan terwyl die netwerk voluit werk | Gemeet 2026-07-28: 4 minute op 8.8% terwyl die xet-log wys grepe beweeg steeds teen ~0.36 MiB/s | Xet laai grepe deurlopend af maar skryf die lêer eers uit sodra 'n hele blok saamgestel is, dus staan die syfer stil en spring dan. Twee gevolge vir 2.3: (a) tempo/ETA word as 'n **kumulatiewe gemiddelde** bereken (reeds so in 2.1a.2) — moenie dit na 'n oombliklike tempo verander nie, dan lees dit as 0.00 MiB/s; (b) **moenie 'n "vasgeval"-waarskuwing op 'n stilstaande persentasie baseer nie** — dit is normale gedrag, nie 'n fout nie. |
| Aflaai teen ~0.6 MiB/s gemeet op Danie se lyn ⇒ ~2.7 uur | Gemeet 2026-07-28 (albei vervoerpaaie eenders) | Die vroeëre "8–41 min"-skatting was optimisties vir hierdie lyn. 2.3/2.4 se tydboodskap moet eerlik wees: dit is 'n "stel op en kom later terug"-taak, net soos die transkripsie self. |
| HF Hub-huisvesting kan later koste/betroubaarheidsdruk kry op skaal | Monitor | Heroorweeg Cloudflare R2 (nul uitgaande-fooie) indien aflaai-volumes/koste dit regverdig. **Bykomende motivering uit 2.1a.2:** R2 oor gewone HTTPS sou ook die Xet-onstabiliteit én die hervattings-gaping (2.1a.3) in een slag oplos. |

---

## Plan-dokument veranderingslog

| Weergawe | Datum | Verandering |
|---|---|---|
| 1.0 | 2026-07-26 | Aanvanklike meesterplan opgestel. Golf 0 afgehandel; Golf 1 (onafhanklike take) vooraan geplaas. |
| 1.1 | 2026-07-26 | Item 1.2 (vertroulikheidsverklaring) afgehandel. Item 1.4 (reproduseerbaarheid) besluit. Golf 2.9 + 2.10 bygevoeg; 5.2 uitgebrei. Waglys opgedateer. |
| 1.2 | 2026-07-27 | Item 1.3 (logo) afgehandel — ware vektor-produksielêers ontvang via Claude Design, geverifieer, twee terugvoer-rondtes. Golf 1 feitlik volledig — slegs 1.1 (André) bly oop. |
| 1.3 | 2026-07-27 | Item 2.1 afgehandel (Opus 5) — Akkuraat-enjin gebou en getoets, 18/18, Fast mode onaangeraak. **Nuwe item 2.1a bygevoeg**: verpakking (transformers/torch/6GB-model) geïdentifiseer as moontlike blokkasie vir 2.3. |
| 1.4 | 2026-07-27 | Item 2.2 afgehandel (Opus 5) — adapter-guard gebou (4 lae), waarde-gebaseerd i.p.v. die meesterplan se oorspronklike (ondoeltreffende) "nie-nul tensor-telling"-voorstel. Agterdeur verwyder; opt-in volledige verifikasie bygevoeg. 2.1a se nota uitgebrei: blokkeer nou ook 2.2 (guard benodig dieselfde afhanklikhede). |
| 1.9 | 2026-07-28 | **Twee los drade toegemaak voor die commit-kontrolepunt.** (A) Weggooi-artefakte van 2.1a.1 verwyder: tak `wave2.1a1/size-investigation` (origin + plaaslik) en konsep-vrystelling `v0.0.0-wave2.1a1-sizetest`. Daar was nooit 'n regte git-tag nie. Geverifieer dat die tak NIE in `main` gemerge was nie voordat dit verwyder is, en dat geen werkvloei daarna verwys nie. **Die tak se inhoud is eers in 2.1a.1 se inskrywing bewaar** — die werklik-gemete afhanklikheidspins (insluitend die onverwagte huggingface_hub-CLI-stapel) sou andersins saam met die tak verdwyn het. (B) Nuwe item **2.1a.4**: modelkaart op die publieke HF-repo (`ac426e10`), met die lisensie by die bron geverifieer (CC-BY-4.0, basismodel Apache-2.0) eerder as uit ons notas aanvaar. Plaaslike commits vir Golf 2.1/2.2/2.1a.2/2.1a.3 geskep vir Danie se hersiening — **niks na origin gestoot nie**. |
| 1.8 | 2026-07-28 | **2.1a.3 gepubliseer en end-tot-end getoets teen die regte repo.** Commit `6baf2473d04da504f039ade149512d891e4a7ca5` — 30 stukke + manifes bygevoeg; monoliet onaangeraak. Oplaai het 5.4 uur gevat teen 0.30 MiB/s; die eerste poging (`upload_folder`, 5 parallelle strome) het ná 1h47m gesterf aan S3-multipart-verbindingsfoute, dus is dit herskryf na `preupload_lfs_files` in bondels van 3 met 2 drade — LFS-dedup het die 8 reeds voltooide stukke (1.64 GiB) behou, en die tweede poging het sonder enige herprobering deurgeloop. **Gemete resultate:** hersamestelling = `33bdc94e…` (bit-identies aan die gepinde model), guard FULL geslaag oor al 1259 tensors; SIGKILL middel-stuk kos net 30 MiB heroordrag i.p.v. 630 MiB (95.2% behou); Xet-terugval werk per stuk; bederfde stuk kos 200 MiB i.p.v. 5888 MiB. **Bug gevang deur die werklike lopie:** die stuk-pad het die lêerpad i.p.v. die modelgids teruggegee — reggestel, met 'n regressietoets wat die kontrak vir albei uitlegte vasspyker. 135/135 toetse. |
| 1.7 | 2026-07-28 | Item **2.1a.3 afgehandel** (Opus 5) — die hervattings-gaping is opgelos deur die model in 30 × ~200 MiB stukke op te deel eerder as deur 'n eie ranged-HTTP-lus te skryf. `shard_accurate_model.py` geskep (greep-vlak opdeling + manifes, plaaslik heen-en-weer geverifieer teen `33bdc94e…`); `accurate_model_download.py` het nou 'n `chunked`-uitleg as verstek, met per-stuk-verifikasie, oorslaan-wat-reeds-daar-is, hersamestelling met heel-lêer-sha256-kontrole, en volledige guard-verifikasie soos voorheen. Stukke oorleef 'n guard-mislukking sodat 'n herprobeer goedkoop is. **Nuwe bevindings op die waglys:** oplaai loop teen ~0.26 MiB/s (≈7 uur, eenmalig, raak net Danie); Xet-deduplisering werk NIE vir greep-snitte nie (29/37 cache misses) en die Xet-oplaaipad tref die bekende verbindingsfoute; skyfbehoefte tydens installasie is ~11.5 GiB. **Opvolg-item bygevoeg:** verwyder die monolitiese lêer (en `layout="single"`) eers ná bevestiging in produksie. |
| 1.6 | 2026-07-28 | Item 2.1a.2 afgehandel (Opus 5) — `accurate_model_download.py` gebou: gepinde revisie `fedc252…`, pad `~/.danscribe_models/accurate-af-large-v3/`, outomatiese Xet-terugval, vorderingsterugroep vir 2.3, onvoorwaardelike **volledige** verifikasie ná aflaai, en uitvee-op-korrupsie. 93/93 toetse. **2.3 se blokkasie opgehef.** Nuwe item 2.1a.2a (Xet-terugval, klaar) en **2.1a.3 (hervatbare aflaai — besluit nodig)** bygevoeg: gemeet dat hf_hub 1.24.0 net per lêer hervat, wat vir hierdie repo beteken die 5.75 GiB-lêer begin ná 'n crash van vooraf; 'n ouer hf_hub is uitgesluit deur transformers se `>=1.5.0`-eis. 2.1a.1 uitgebrei met die `requirements.txt`-pins wat nou nodig is. Waglys uitgebrei met vier gemete bevindings (werklike grootte 5.76 GiB nie 3.69 GiB nie; hervattings-gaping; Xet se sprongsgewyse vordering; ~2.7 uur werklike aflaaityd). |
| 1.5 | 2026-07-27 | Item 2.1a ondersoek en besluit (Opus 5 ondersoek): torch reeds in exe; ware delta ~36–40 MiB. **BESLUIT: aflaai-op-aanvraag via Hugging Face Hub.** Nuwe items 2.1a.1 (empiriese CI-validasie) en 2.1a.2 (bou aflaai-meganisme) bygevoeg. **Nuwe item 2.2a bygevoeg**: verstel volledige verifikasie na verstek — Danie het ingestem, nog nie geïmplementeer nie. Waglys uitgebrei met aflaai-verwante risiko's. |

---

*Hierdie plan leef in die repo en word met git ge-versiebeheer. Elke opdatering
= 'n commit. Werk die status-kolomme en die veranderingslog by soos werk vorder.*
