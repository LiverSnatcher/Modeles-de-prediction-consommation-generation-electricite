# Modeles-de-prediction-consommation-et-generation-electricite
# Réseau Électrique · France

Prévision de la consommation, de la production éolienne et solaire, et de la charge résiduelle
du réseau électrique français. Données temps réel RTE via l'API éCO2mix (ODRE).

**Démo :** https://modeles-de-prediction-consommation-generation-electricite-hk64.streamlit.app

---

## Comment ça marche

Trois modèles indépendants tournent en parallèle :

- **Consommation** — XGBoost entraîné sur les données RTE 2015–2025 (30 min).
  Features : heure, jour, mois, jours fériés, température Open-Meteo, lags et moyennes glissantes.

![Consommation](screenshots/consommation.png)

- **Solaire** — XGBoost, la cible est un load factor (0–1) reconverti en MW, avec météo multi-zones
  sur 4 régions (Occitanie, PACA, Nouvelle-Aquitaine, Auvergne-RA). MAE = 301 MW.

![Solaire](screenshots/solaire.png)

- **Éolien** — Temporal Fusion Transformer (PyTorch Forecasting, ~1M params), horizon H+24h,
  7 zones météo à 100m d'altitude, courbe de puissance physique + correction densité de l'air.
  MAPE = 14,7% — c'est le modèle le moins fiable des trois, l'éolien étant difficile à prévoir
  au-delà de quelques heures. Les prévisions à horizon long peuvent diverger.

![Eolien](screenshots/eolien.png)

- **Charge résiduelle** — calculée comme Consommation − Éolien − Solaire.

![Charge résiduelle](screenshots/charge%20residuelle.png)

Les résultats sont mis en cache dans **Supabase** et rafraîchis toutes les heures.
Si la dernière prévision date de moins d'1h, elle est chargée directement depuis la DB
sans relancer les modèles.

---

## Stack

Python · XGBoost · PyTorch Forecasting · Lightning · Streamlit · Plotly · Supabase · Open-Meteo

---

## Lancer en local

```bash
git clone https://github.com/LiverSnatcher/modeles-de-prediction-consommation-generation-electricite
cd modeles-de-prediction-consommation-generation-electricite
pip install -r requirements.txt
streamlit run app2.py
```

Ajouter un fichier `.streamlit/secrets.toml` :
```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
```

Trois fichiers modèles attendus à la racine : `model_consommation.pkl`, `model_solaire_v2.pkl`, `model_eolien_tft.ckpt`.

---

Maxime Mbende — Bachelor IA & Data, Nexa Digital School (alternance)  
[LinkedIn](https://www.linkedin.com/in/maxime-mbende) · [GitHub](https://github.com/LiverSnatcher)
