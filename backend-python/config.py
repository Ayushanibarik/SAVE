"""
Configuration Management for Disaster Response System
Environment-based settings with sensible defaults
"""

import os


class Config:
    """Application configuration with environment variable overrides"""

    # Flask
    DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))

    # Data Sources
    DATA_SOURCE = os.getenv("DATA_SOURCE", "osm")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = os.getenv("MONGODB_DB", "disaster_response")

    # Google Sheets
    GOOGLE_SHEETS_HOSPITALS_URL = os.getenv("GOOGLE_SHEETS_HOSPITALS_URL", "")
    GOOGLE_SHEETS_AMBULANCES_URL = os.getenv("GOOGLE_SHEETS_AMBULANCES_URL", "")

    # Caching
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", 1800))  # 30 minutes
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 128))

    # API Limits
    OSM_TIMEOUT = int(os.getenv("OSM_TIMEOUT", 20))
    OSM_MAX_RETRIES = int(os.getenv("OSM_MAX_RETRIES", 3))
    MAX_PATIENTS = int(os.getenv("MAX_PATIENTS", 500))
    MAX_RADIUS_KM = int(os.getenv("MAX_RADIUS_KM", 50))
    DEFAULT_RADIUS_KM = int(os.getenv("DEFAULT_RADIUS_KM", 10))

    # Agent Messages
    MAX_AGENT_MESSAGES = int(os.getenv("MAX_AGENT_MESSAGES", 100))

    # Analytics
    ANALYTICS_DB = os.getenv("ANALYTICS_DB", "disaster_analytics.db")

    # ESI Triage
    ESI_ENABLED = os.getenv("ESI_ENABLED", "true").lower() == "true"

    # Ambulance ETA
    AMBULANCE_SPEED_KM_PER_MIN = float(os.getenv("AMBULANCE_SPEED", 0.67))  # ~40km/h in urban

    # Hospital Capacity Estimation (when beds unknown from OSM)
    DEFAULT_SMALL_HOSPITAL_BEDS = 15
    DEFAULT_MEDIUM_HOSPITAL_BEDS = 50
    DEFAULT_LARGE_HOSPITAL_BEDS = 150

    # ============================================
    # AI/ML ENGINE CONFIGURATION
    # ============================================

    # Deep Learning
    DL_SEVERITY_LAYERS = [22, 64, 128, 64, 32, 5]     # MLP architecture
    DL_DEMAND_LAYERS = [12, 32, 64, 32, 4]
    DL_DROPOUT_RATE = float(os.getenv("DL_DROPOUT", 0.15))
    DL_MC_DROPOUT_SAMPLES = int(os.getenv("DL_MC_SAMPLES", 20))

    # Reinforcement Learning
    RL_LEARNING_RATE = float(os.getenv("RL_LR", 0.001))
    RL_GAMMA = float(os.getenv("RL_GAMMA", 0.99))
    RL_EPSILON_START = float(os.getenv("RL_EPSILON", 0.3))
    RL_REPLAY_BUFFER_SIZE = int(os.getenv("RL_BUFFER", 10000))
    RL_BATCH_SIZE = int(os.getenv("RL_BATCH", 32))

    # Graph Neural Network
    GNN_HIDDEN_DIM = int(os.getenv("GNN_HIDDEN", 32))
    GNN_NUM_LAYERS = int(os.getenv("GNN_LAYERS", 3))
    GNN_ATTENTION_HEADS = int(os.getenv("GNN_HEADS", 4))

    # Multi-Objective Optimization (NSGA-II)
    NSGA_POPULATION = int(os.getenv("NSGA_POP", 100))
    NSGA_GENERATIONS = int(os.getenv("NSGA_GENS", 50))
    NSGA_CROSSOVER_PROB = float(os.getenv("NSGA_CX", 0.9))
    NSGA_MUTATION_PROB = float(os.getenv("NSGA_MUT", 0.1))

    # Markov Decision Process
    MDP_GAMMA = float(os.getenv("MDP_GAMMA", 0.95))
    MDP_MC_SIMULATIONS = int(os.getenv("MDP_SIMS", 1000))
    MDP_TIME_HORIZON_HOURS = int(os.getenv("MDP_HORIZON", 48))

    # NLP Clinical Reasoning
    NLP_ATTENTION_HEADS = int(os.getenv("NLP_HEADS", 4))
    NLP_FEATURE_DIM = int(os.getenv("NLP_DIM", 16))

    # Multi-Agent Reinforcement Learning (MARL)
    MARL_COMM_DIM = int(os.getenv("MARL_COMM_DIM", 16))
    MARL_MIXING_DIM = int(os.getenv("MARL_MIX_DIM", 32))
    MARL_HIDDEN_DIM = int(os.getenv("MARL_HIDDEN", 64))
    MARL_EPSILON = float(os.getenv("MARL_EPSILON", 0.3))
    MARL_GAMMA = float(os.getenv("MARL_GAMMA", 0.99))
