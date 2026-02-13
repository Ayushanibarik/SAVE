# S.A.V.E. — Strategic Agent-based Victim Evacuation System

🚨 **AI-powered disaster response system** with multi-agent coordination, real-time optimization, and emergency voice calls.

## 🎯 Features

- **7 AI Models**: Deep Learning, Reinforcement Learning, MARL, GNN, NSGA-II, MDP, NLP
- **Real-time Coordination**: Multi-agent system optimizes patient allocation across hospitals
- **AI Voice Calls**: Twilio integration for automated emergency notifications
- **ESI Triage System**: Medical-grade patient classification
- **Live GIS Mapping**: OpenStreetMap integration with heatmaps and risk zones
- **Real Data Sources**: FEMA, WHO, NDMA disaster datasets

## 🚀 Quick Start (Local)

### Backend
```bash
cd backend-python
pip install -r requirements.txt
python optimize.py
```

### Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Visit: http://localhost:8501

## ☁️ Cloud Deployment

See [`DEPLOY.md`](DEPLOY.md) for complete deployment instructions.

**Backend:** PythonAnywhere (free forever)  
**Frontend:** Streamlit Cloud (free)

## 📞 Twilio Setup

1. Sign up at [twilio.com](https://www.twilio.com/try-twilio)
2. Get free $15 credit
3. Set environment variables:
   ```bash
   TWILIO_ACCOUNT_SID=ACxxxxxx
   TWILIO_AUTH_TOKEN=xxxxxx
   TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
   ```

## 📊 System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│  Dashboard  │────▶│  Backend API │────▶│  Twilio  │
│  (Streamlit)│     │   (Flask)    │     │   Voice  │
└─────────────┘     └──────────────┘     └──────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ OpenStreetMap│
                    │  (Nominatim) │
                    └──────────────┘
```

## 🧠 AI Models

| Model | Purpose |
|:---|:---|
| **Deep Learning (MLP)** | Severity prediction & demand forecasting |
| **Reinforcement Learning (DQN)** | Dynamic resource allocation |
| **MARL** | Multi-agent hospital coordination |
| **Graph Neural Network** | Network flow optimization |
| **NSGA-II** | Multi-objective Pareto optimization |
| **Markov Decision Process** | Sequential decision making |
| **NLP Clinical Agent** | Medical reasoning & triage |

## 📁 Project Structure

```
├── backend-python/          # Flask API backend
│   ├── optimize.py         # Main API server
│   ├── ai_caller.py        # Twilio voice integration
│   ├── rl_optimizer.py     # RL/MARL agents
│   └── requirements.txt
├── dashboard/              # Streamlit frontend
│   ├── app.py             # Main dashboard
│   └── requirements.txt
└── DEPLOY.md              # Deployment guide
```

## 📜 License

© 2026 Emergency Response AI Division

---

**Built with:** Python • Flask • Streamlit • PyTorch • Twilio • OpenStreetMap
