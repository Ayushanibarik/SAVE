# n8n Multi-Agent Workflow Setup Guide

Complete step-by-step guide for setting up all 4 agent workflows in n8n.

---

## Prerequisites

1. Install n8n: `npm install -g n8n`
2. Start n8n: `npx n8n`
3. Open: `http://localhost:5678`

---

## Workflow 1: Disaster Trigger Agent (Main Entry)

This is the main entry point that triggers all other agents.

### Step 1: Create New Workflow
- Click **New Workflow** → Name it: `Disaster Trigger Agent`

### Step 2: Add Webhook Node
| Setting | Value |
|---------|-------|
| Method | POST |
| Path | `disaster-trigger` |

**Test URL:** `http://localhost:5678/webhook/disaster-trigger`

**Test Input:**
```json
{
  "lat": 22.721,
  "lng": 88.485,
  "patients": 6
}
```

### Step 3: Add Set Node (Metadata)
Add fields:
- `severity` = `HIGH`
- `type` = `FIRE`

### Step 4: Add HTTP Request → Hospital Agent
| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://localhost:5678/webhook/hospital-data` |
| Body Type | JSON |

Body:
```json
{
  "lat": {{ $json.lat }},
  "lng": {{ $json.lng }},
  "patients": {{ $json.patients }}
}
```

### Step 5: Add HTTP Request → Emergency Agent
| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://localhost:5678/webhook/ambulance-data` |
| Body Type | JSON |

Same body as above.

### Connection Flow:
```
Webhook → Set Node → HTTP Request (Hospital)
                   → HTTP Request (Emergency)
```

---

## Workflow 2: Hospital Agent

### Step 1: Create New Workflow
- Name: `Hospital Agent`

### Step 2: Add Webhook Node
| Setting | Value |
|---------|-------|
| Method | POST |
| Path | `hospital-data` |

### Step 3: Add Set Node (Hospital Database)
Use **Item Lists** node to create multiple items:

| Hospital | Beds | lat | lng |
|----------|------|-----|-----|
| A | 3 | 22.722 | 88.481 |
| B | 5 | 22.726 | 88.490 |
| C | 2 | 22.718 | 88.478 |

### Step 4: Add HTTP Request → Python Backend
| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://localhost:5000/optimize` |
| Body Type | JSON |

Body:
```json
{
  "patients": {{ $node["Webhook"].json.patients }},
  "lat": {{ $node["Webhook"].json.lat }},
  "lng": {{ $node["Webhook"].json.lng }},
  "hospitals": {{ $json }}
}
```

### Step 5: Add HTTP Request → Government Agent
| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://localhost:5678/webhook/decision` |

---

## Workflow 3: Emergency Agent

### Step 1: Create New Workflow
- Name: `Emergency Agent`

### Step 2: Add Webhook Node
| Setting | Value |
|---------|-------|
| Path | `ambulance-data` |

### Step 3: Add Set Node (Ambulance Data)
Add fields:
- Ambulance 1: `22.720,88.480`
- Ambulance 2: `22.724,88.487`

### Step 4: Add HTTP Request → Government Agent
| Setting | Value |
|---------|-------|
| Method | POST |
| URL | `http://localhost:5678/webhook/decision` |

---

## Workflow 4: Government Decision Agent

### Step 1: Create New Workflow
- Name: `Government Decision Agent`

### Step 2: Add Webhook Node
| Setting | Value |
|---------|-------|
| Path | `decision` |

### Step 3: Add Merge Node
- Combine: Hospital Allocation + Ambulance Data

### Step 4: Add Respond to Webhook Node
Returns final JSON with allocation result.

---

## Testing the Complete Flow

1. **Start Python Backend:**
   ```bash
   cd backend-python
   python optimize.py
   ```

2. **Activate All Workflows** in n8n (toggle ON)

3. **Test via cURL:**
   ```bash
   curl -X POST http://localhost:5678/webhook/disaster-trigger \
     -H "Content-Type: application/json" \
     -d '{"lat":22.721,"lng":88.485,"patients":6}'
   ```

4. **Or use Streamlit Dashboard:**
   ```bash
   cd dashboard
   streamlit run app.py
   ```
   Click "Trigger Disaster Response"

---

## Architecture Diagram

```
      ┌─────────────────────────────────────────────────────────┐
      │                    STREAMLIT DASHBOARD                   │
      │                  (User Interface + Map)                  │
      └──────────────────────────┬──────────────────────────────┘
                                 │ POST /disaster-trigger
                                 ▼
      ┌─────────────────────────────────────────────────────────┐
      │              DISASTER TRIGGER AGENT (n8n)               │
      │              Webhook + Set Metadata Node                │
      └────────────────┬────────────────────┬───────────────────┘
                       │                    │
         POST /hospital-data      POST /ambulance-data
                       │                    │
                       ▼                    ▼
      ┌────────────────────────┐  ┌────────────────────────────┐
      │   HOSPITAL AGENT       │  │   EMERGENCY AGENT          │
      │   n8n Workflow         │  │   n8n Workflow             │
      └───────────┬────────────┘  └─────────────┬──────────────┘
                  │                              │
    POST /optimize│                              │
                  ▼                              │
      ┌────────────────────────┐                 │
      │   PYTHON BACKEND       │                 │
      │   Flask Optimization   │                 │
      └───────────┬────────────┘                 │
                  │                              │
                  ▼                              ▼
      ┌─────────────────────────────────────────────────────────┐
      │           GOVERNMENT DECISION AGENT (n8n)               │
      │           Merge Results + Final Decision                │
      └─────────────────────────────────────────────────────────┘
```
