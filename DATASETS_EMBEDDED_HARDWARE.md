# Datasets Embedded Systems, Hardware & Bus Protocols
## Recherche exhaustive pour Micro_KIKI - Avril 2026

---

## SYNTHESE RAPIDE

| Domaine | Datasets trouvees | Niveau de rarete |
|---------|------------------|-----------------|
| Embedded/Firmware (ESP-IDF, STM32) | 8 datasets | TRES RARE |
| Arduino/Microcontroller | 6 datasets | RARE |
| RTOS (FreeRTOS, Zephyr) | 0 dataset direct | DESERT |
| Bus Protocols (I2C/SPI/CAN/UART) | 0 dataset code | DESERT |
| BLE/MQTT/LoRa/Zigbee | 0 dataset code | DESERT |
| KiCad/PCB design | 1 dataset (le notre) | DESERT |
| SPICE/Circuit simulation | 3 datasets (recherche) | RARE |
| Analog circuit design | 5 datasets (recherche) | RARE-MOYEN |
| Verilog/HDL (adjacent) | 20+ datasets | MOYEN |
| PlatformIO/CMake/Build | 0 dataset | DESERT |
| Electronics datasheets | 0 dataset structure | DESERT |

**Conclusion: ces domaines sont parmi les plus sous-representes sur HuggingFace. Le scraping GitHub est OBLIGATOIRE pour creer les datasets.**

---

## 1. DATASETS EXISTANTS TROUVES

### A. Embedded & Firmware

#### 1. MuratKomurcu/stm32-hal-dataset
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: Parquet (text-generation)
- **Downloads**: 76
- **Licence**: MIT
- **Description**: Code STM32 HAL reel collecte depuis GitHub, 11 categories de peripheriques, multiples familles STM32
- **Modele associe**: `MuratKomurcu/starcoder2-stm32`
- **Lien**: https://huggingface.co/datasets/MuratKomurcu/stm32-hal-dataset
- **PRIORITE HAUTE pour Micro_KIKI**

#### 2. gouthamsk/esp_idf_code
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: Parquet (text)
- **Downloads**: 11
- **Description**: Code ESP-IDF collecte
- **Lien**: https://huggingface.co/datasets/gouthamsk/esp_idf_code
- **PRIORITE HAUTE pour Micro_KIKI**

#### 3. tgrlkk/stm32-hal-mistral
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: Parquet
- **Downloads**: 4
- **Licence**: MIT
- **Lien**: https://huggingface.co/datasets/tgrlkk/stm32-hal-mistral

#### 4. ZiHDeng/hf-stm32-v1
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: Parquet
- **Downloads**: 6
- **Licence**: Apache 2.0
- **Lien**: https://huggingface.co/datasets/ZiHDeng/hf-stm32-v1

#### 5. Ailiance-fr/mascarade-stm32-dataset (NOTRE DATASET)
- **Source**: HuggingFace
- **Taille**: 1K-10K exemples
- **Format**: JSON
- **Downloads**: 10
- **Lien**: https://huggingface.co/datasets/Ailiance-fr/mascarade-stm32-dataset

#### 6. Ailiance-fr/kill-life-embedded-qa (NOTRE DATASET)
- **Source**: HuggingFace
- **Taille**: <1K exemples
- **Format**: JSON
- **Downloads**: 51
- **Lien**: https://huggingface.co/datasets/Ailiance-fr/kill-life-embedded-qa

#### 7. forlinx-embedded/forlinx-downloads
- **Source**: HuggingFace
- **Taille**: indeterminee
- **Downloads**: 382
- **Tags**: embedded-hardware
- **Lien**: https://huggingface.co/datasets/forlinx-embedded/forlinx-downloads

#### 8. cagatayakinucar/microprocessors_and_microcontrollers_dataset
- **Source**: HuggingFace
- **Taille**: 1K-10K exemples
- **Format**: JSON (text)
- **Downloads**: 3
- **Lien**: https://huggingface.co/datasets/cagatayakinucar/microprocessors_and_microcontrollers_dataset

### B. Arduino / Microcontroller

#### 9. g4lihru/arduino-dataset
- **Source**: HuggingFace
- **Taille**: 100M-1B tokens (ENORME)
- **Format**: texte
- **Downloads**: 267
- **Licence**: MIT
- **Tags**: code-generation, gpt
- **Lien**: https://huggingface.co/datasets/g4lihru/arduino-dataset
- **PRIORITE HAUTE - Le plus gros dataset embedded trouve**

#### 10. CJJones/Multiturn_Microcontroller-Arduino-LLM-Training
- **Source**: HuggingFace
- **Taille**: 100K-1M exemples
- **Format**: Text (multiturn conversations)
- **Downloads**: 40
- **Licence**: CC-BY-NC-4.0
- **Description**: Conversations multitour pour entrainer un LLM sur microcontroleurs/Arduino
- **Lien**: https://huggingface.co/datasets/CJJones/Multiturn_Microcontroller-Arduino-LLM-Training
- **PRIORITE HAUTE - Format instruction ideal**

#### 11. suneeldk/arduino-code-dataset
- **Source**: HuggingFace
- **Taille**: <1K exemples
- **Format**: JSON (text-generation)
- **Downloads**: 70
- **Licence**: MIT
- **Lien**: https://huggingface.co/datasets/suneeldk/arduino-code-dataset

#### 12. gavmac00/arduino-docs
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: JSON
- **Downloads**: 36
- **Licence**: CC-BY-3.0
- **Lien**: https://huggingface.co/datasets/gavmac00/arduino-docs

#### 13. bshada/arduino.stackexchange.com
- **Source**: HuggingFace
- **Taille**: 10K-100K exemples
- **Format**: JSON (Q&A StackExchange)
- **Downloads**: 26
- **Licence**: CC-BY-SA-3.0
- **Lien**: https://huggingface.co/datasets/bshada/arduino.stackexchange.com
- **BON POUR Q&A embedde**

#### 14. Telles1974/Arduino-1000
- **Source**: HuggingFace
- **Taille**: non specifiee
- **Format**: Q&A text-generation
- **Downloads**: 26
- **Licence**: CC-BY-4.0
- **Lien**: https://huggingface.co/datasets/Telles1974/Arduino-1000

### C. KiCad / PCB

#### 15. Ailiance-fr/mascarade-kicad-dataset (NOTRE DATASET)
- **Source**: HuggingFace
- **Taille**: 1K-10K exemples
- **Format**: JSON
- **Downloads**: 14
- **Lien**: https://huggingface.co/datasets/Ailiance-fr/mascarade-kicad-dataset

*Aucun autre dataset KiCad structure n'existe sur HuggingFace.*

### D. SPICE / Circuit Simulation

#### 16. Masala-CHAI (Auto-SPICE)
- **Source**: GitHub + Google Drive
- **Taille**: ~7,500 SPICE netlists extraits de 10 textbooks
- **Format**: SPICE netlists + metadata
- **Description**: Framework automatise LLM pour extraction de netlists SPICE depuis schemas de circuits analogiques
- **Papier**: arXiv 2411.14299
- **GitHub**: https://github.com/jitendra-bhandari/Masala-CHAI
- **Download**: https://drive.google.com/file/d/1aNC-8mye_Pbw9nYS0cmN2ggUaThJHUP2
- **PRIORITE HAUTE pour spice-life**

#### 17. SPICEPilot
- **Source**: GitHub
- **Format**: Python-based SPICE code generation benchmarks (Easy/Medium/Hard/Extreme)
- **Licence**: MIT
- **GitHub**: https://github.com/ACADLab/SPICEPilot
- **Description**: Framework de benchmarking SPICE avec generation de code, validation, evaluation
- **UTILE pour benchmark, pas directement pour training**

#### 18. SynC-LLM (SynCircuitData)
- **Source**: GitHub
- **Format**: Synthetic digital circuit code (Verilog)
- **Description**: Generation de circuits synthetiques a grande echelle via diffusion + LLM
- **Papier**: EMNLP 2025
- **GitHub**: https://github.com/hkust-zhiyao/SynCircuitData
- **UTILE pour circuits digitaux**

#### 19. AMSnet-KG
- **Source**: Academique
- **Format**: Netlists + Knowledge Graph + annotations
- **Description**: Dataset de circuits analogiques/mixed-signal avec graphe de connaissances
- **Site**: https://ams-net.github.io/
- **Papier**: arXiv 2411.13560 / ACM TODAES
- **PRIORITE HAUTE si accessible**

#### 20. Netlist datasets sur HF
- `Vrindarani/netlistgen` (13 dl)
- `zeju-0727/ForgeEDA_netlist` (10 dl)
- `goihere/Netlist_data` (6 dl)
- `perkros/netlist-snippets-20-lines` (5 dl)
- `perkros/netlist-snippets-40-lines` (4 dl)
- `perkros/netlist-snippets-80-lines` (5 dl)
- `perkros/netlist-metadata` (5 dl)

### E. Analog Circuit Design (LLM-specific)

#### 21. AnalogSeeker (QTSA dataset)
- **Source**: HuggingFace (model) + dataset inclus
- **Taille**: 112.65M tokens
- **Description**: Foundation LLM pour conception circuits analogiques, dataset construit par distillation de connaissances multi-agent
- **Modele**: https://huggingface.co/analogllm/analogseeker
- **Performance**: 85.04% accuracy sur AMSBench-TQA
- **PRIORITE HAUTE - dataset de reference pour analogique**

#### 22. CIRCUIT Benchmark
- **Source**: Academique
- **Taille**: 510 Q&A pairs
- **Description**: Benchmark pour evaluation des capacites de raisonnement circuit analogique des LLMs
- **Papier**: arXiv 2502.07980
- **UTILE pour evaluation**

#### 23. englund/circuit-design-benchmarks
- **Source**: HuggingFace
- **Taille**: <1K exemples
- **Format**: Parquet
- **Downloads**: 4
- **Lien**: https://huggingface.co/datasets/englund/circuit-design-benchmarks

### F. Verilog/HDL (Adjacent - utile pour digital design)

| Dataset | Downloads | Taille | Lien |
|---------|-----------|--------|------|
| shailja/Verilog_GitHub | 391 | 100K-1M | https://huggingface.co/datasets/shailja/Verilog_GitHub |
| verify-ppt/marin-starcoderdata_verilog | 407 | ? | HF |
| bnadimi/PyraNet-Verilog | 309 | 100K-1M | https://huggingface.co/datasets/bnadimi/PyraNet-Verilog |
| GaTech-EIC/MG-Verilog | 197 | ? | https://huggingface.co/datasets/GaTech-EIC/MG-Verilog |
| dakies/nvlabs-verilogeval | 173 | <1K | https://huggingface.co/datasets/dakies/nvlabs-verilogeval |
| dakies/nvlabs-verilogeval-v2-spec-to-rtl | 159 | <1K | HF |
| JayZhang1/Verilogdata4pretrainCODET5 | 146 | 100K-1M | HF |
| NOKHAB-Lab/LLM_4_Verilog | 76 | 10K-100K | HF |
| davide221/verilog-instruct-60k | 51 | 10K-100K | HF |
| AbiralArch/verilog-training-data | 18 | 1K-10K | HF |

---

## 2. REPOS GITHUB A SCRAPER POUR CREER DES DATASETS

### Priorite 1: Firmware & RTOS (DESERT TOTAL)

| Repo | Stars | Contenu scrapeable | Volume estime |
|------|-------|-------------------|---------------|
| **espressif/esp-idf** | 17,831 | `examples/` (200+ exemples complets), `components/` | ~500K lignes C |
| **espressif/esp-idf/examples** | inclus | Chaque sous-dossier = 1 exemple complet avec CMake | ~200 exemples |
| **espressif/arduino-esp32** | ~14K | Libraries, examples | ~200K lignes |
| **zephyrproject-rtos/zephyr** | 14,992 | `samples/` (400+ exemples), `drivers/` | ~1M lignes C |
| **FreeRTOS/FreeRTOS** | 7,209 | `FreeRTOS/Demo/` (50+ demos), kernel source | ~200K lignes C |
| **STMicroelectronics/STM32CubeF4** | ~1K | `Projects/*/Examples/` | ~300K lignes C |
| **STMicroelectronics/STM32CubeH7** | ~800 | `Projects/*/Examples/` | ~400K lignes C |
| **STMicroelectronics/STM32CubeL4** | ~500 | Meme structure | ~250K lignes C |
| **STMicroelectronics/STM32CubeG4** | ~300 | Meme structure | ~200K lignes C |
| **apache/nuttx** | ~9K | RTOS complet, drivers, examples | ~500K lignes C |
| **RT-Thread/rt-thread** | ~11K | RTOS + BSP + packages | ~800K lignes C |
| **RIOT-OS/RIOT** | ~5K | `examples/`, `drivers/`, `tests/` | ~400K lignes C |

### Priorite 2: Bus Protocols (ZERO datasets existants)

| Repo | Stars | Protocoles couverts |
|------|-------|-------------------|
| **espressif/esp-idf/examples/peripherals** | inclus | I2C, SPI, UART, TWAI(CAN), SDMMC, USB |
| **torvalds/linux (drivers/)** | 193K | Tous les drivers I2C/SPI/UART/CAN/USB du kernel |
| **linux-can/can-utils** | ~2.5K | Outils CAN bus userspace |
| **collin80/esp32_can** | ~400 | CAN bus ESP32 |
| **sandeepmistry/arduino-CAN** | ~800 | CAN bus Arduino |
| **nopnop2002/esp-idf-can2mqtt** | ~100 | CAN-to-MQTT bridge ESP-IDF |
| **adafruit/Adafruit_BusIO** | ~300 | I2C/SPI abstractions Arduino |
| **stm32duino/Arduino_Core_STM32** | ~3K | HAL I2C/SPI/UART/CAN wrappers |

### Priorite 3: BLE / MQTT / LoRa / Zigbee

| Repo | Stars | Protocole |
|------|-------|-----------|
| **espressif/esp-idf/examples/bluetooth** | inclus | BLE GATT, GAP, A2DP, SPP |
| **apache/mynewt-nimble** | ~700 | BLE stack complet |
| **eclipse/mosquitto** | ~10K | MQTT broker reference |
| **eclipse/paho.mqtt.c** | ~2K | MQTT client C |
| **knolleary/pubsubclient** | ~4K | MQTT Arduino |
| **Lora-net/LoRaMac-node** | ~2K | LoRaWAN stack reference |
| **mcci-catena/arduino-lmic** | ~800 | LoRaWAN Arduino |
| **Koenkk/zigbee2mqtt** | ~13K | Zigbee stack |
| **project-chip/connectedhomeip** | ~8K | Matter protocol stack |

### Priorite 4: KiCad / PCB / Schematiques

| Repo | Stars | Contenu |
|------|-------|---------|
| **KiCad/kicad-symbols** | ~1.5K | Toutes les librairies de symboles KiCad |
| **KiCad/kicad-footprints** | ~1.5K | Toutes les empreintes |
| **KiCad/kicad-templates** | ~300 | Templates de projets |
| **sparkfun/SparkFun-KiCad-Libraries** | ~300 | Librairies SparkFun |
| **Digi-Key/digikey-kicad-library** | ~2K | Librairie Digi-Key |
| **adafruit/Adafruit-Eagle-Library** | ~1K | Composants Adafruit (Eagle, convertible) |
| **OLIMEX/OLINUXINO** | ~500 | Projets KiCad complets open-source |
| **wickerbox/wickerlib** | ~100 | KiCad library |
| Divers repos "Open Hardware" | variable | Projets KiCad complets |

### Priorite 5: SPICE / Simulation

| Repo | Stars | Contenu |
|------|-------|---------|
| **jitendra-bhandari/Masala-CHAI** | ~50 | Pipeline extraction SPICE |
| **ACADLab/SPICEPilot** | ~30 | Benchmarks SPICE |
| **hkust-zhiyao/SynCircuitData** | ~20 | Circuits synthetiques |
| **ngspice/ngspice** | ~500 | Exemples dans `examples/` |
| **ltspice (collections)** | divers | Modeles et exemples LTSpice |

### Priorite 6: PlatformIO / Build Systems

| Repo | Stars | Contenu |
|------|-------|---------|
| **platformio/platformio-examples** | ~500 | Exemples multi-framework |
| **platformio/platform-espressif32** | ~1K | Configs ESP32 |
| **platformio/platform-ststm32** | ~500 | Configs STM32 |

---

## 3. STRATEGIE DE SCRAPING RECOMMANDEE

### Phase 1: Datasets existants a telecharger IMMEDIATEMENT

```bash
# Haute priorite
pip install datasets
python -c "
from datasets import load_dataset
# STM32 HAL
ds = load_dataset('MuratKomurcu/stm32-hal-dataset')
ds.save_to_disk('data/stm32-hal')
# Arduino mega dataset
ds = load_dataset('g4lihru/arduino-dataset')
ds.save_to_disk('data/arduino-mega')
# Multiturn microcontroller
ds = load_dataset('CJJones/Multiturn_Microcontroller-Arduino-LLM-Training')
ds.save_to_disk('data/multiturn-mcu')
# ESP-IDF code
ds = load_dataset('gouthamsk/esp_idf_code')
ds.save_to_disk('data/esp-idf-code')
# Arduino docs
ds = load_dataset('gavmac00/arduino-docs')
ds.save_to_disk('data/arduino-docs')
# Arduino StackExchange
ds = load_dataset('bshada/arduino.stackexchange.com')
ds.save_to_disk('data/arduino-stackexchange')
# Verilog (pour digital/HDL)
ds = load_dataset('shailja/Verilog_GitHub')
ds.save_to_disk('data/verilog-github')
"
```

### Phase 2: Scraping GitHub pour creer des datasets (CRITIQUE)

**ESP-IDF examples** (~200 exemples structurees):
```bash
git clone --depth 1 https://github.com/espressif/esp-idf.git /tmp/esp-idf
# Scraper examples/ -> paires (README.md description + main/*.c code)
```

**Zephyr samples** (~400 exemples):
```bash
git clone --depth 1 https://github.com/zephyrproject-rtos/zephyr.git /tmp/zephyr
# Scraper samples/ -> paires (README.rst + src/*.c)
```

**STM32Cube HAL** (tous les peripheriques):
```bash
for family in F4 H7 L4 G4 F7 L0 F1 F0 G0 WB WL U5; do
  git clone --depth 1 "https://github.com/STMicroelectronics/STM32Cube${family}.git" "/tmp/stm32cube-${family}"
done
# Scraper Projects/*/Examples/ -> structure par peripherique
```

**FreeRTOS demos**:
```bash
git clone --depth 1 https://github.com/FreeRTOS/FreeRTOS.git /tmp/freertos
# Scraper FreeRTOS/Demo/ et FreeRTOS-Plus/Demo/
```

**Linux kernel drivers** (MASSIF):
```bash
git clone --depth 1 https://github.com/torvalds/linux.git /tmp/linux
# Scraper drivers/i2c/ drivers/spi/ drivers/net/can/ drivers/usb/ drivers/gpio/
# Volume: ~2M lignes de code driver
```

### Phase 3: Datasets synthetiques a generer

Pour les domaines DESERT (bus protocols, build configs, datasheets), generer des datasets synthetiques:

1. **Bus Protocol Q&A**: Generer des paires instruction/code pour I2C/SPI/CAN/UART en utilisant les exemples scrapes comme seed
2. **PlatformIO configs**: Scraper tous les `platformio.ini` de GitHub, generer des paires description/config
3. **CMakeLists.txt embedded**: Extraire de esp-idf/zephyr, generer des paires
4. **KiCad schematics**: Convertir nos 22 design blocks makelife-hard en paires description/netlist

---

## 4. REFERENCES ACADEMIQUES CLES

| Papier | Annee | Dataset/Tool | Domaine |
|--------|-------|-------------|---------|
| Masala-CHAI | 2024 | 7,500 SPICE netlists | Analog circuits |
| AMSnet-KG | 2024 | AMS netlists + KG | Mixed-signal |
| AnalogSeeker | 2025 | 112.65M tokens QTSA | Analog LLM |
| SynC-LLM | 2025 | Synthetic circuits | Digital circuits |
| SPICEPilot | 2024 | SPICE benchmarks | Simulation |
| CircuitLM | 2025 | CircuitJSON pipeline | Schematics |
| SPICEAssistant | 2025 | SMPS design | Power supply |
| CIRCUIT benchmark | 2025 | 510 Q&A | Evaluation |
| Awesome-LLM4EDA | ongoing | Survey+resources | EDA survey |
| VerilogEval v2 | 2024 | RTL benchmarks | Verilog |
| RTLLM 2.0 | 2024 | 50 RTL designs | Verilog |

**Repo de reference**: https://github.com/Thinklab-SJTU/Awesome-LLM4EDA

---

## 5. ESTIMATION DE VOLUME TOTAL SCRAPEABLE

| Source | Volume estime | Effort scraping |
|--------|--------------|----------------|
| ESP-IDF examples | ~50K paires | 1 jour |
| Zephyr samples | ~80K paires | 1 jour |
| STM32Cube (toutes familles) | ~200K paires | 2 jours |
| FreeRTOS demos | ~20K paires | 0.5 jour |
| Linux kernel drivers | ~500K paires | 3 jours |
| Arduino libraries | ~100K paires | 1 jour |
| KiCad symbols/footprints | ~30K entries | 1 jour |
| Bus protocol examples | ~10K paires | 1 jour |
| BLE/MQTT/LoRa stacks | ~50K paires | 2 jours |
| StackExchange embedded | ~50K Q&A | 1 jour |
| **TOTAL** | **~1.1M paires** | **~13 jours** |

---

*Genere le 2026-04-15 par recherche exhaustive HuggingFace API + WebSearch + GitHub CLI*
